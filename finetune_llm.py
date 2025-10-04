from config.base import toxicity_labels, train_file, test_file, test_labels_file
from src import load_data, prepare_dataset, load_config
from src.utils import timer
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
import torch
from transformers import AdamW, BertForSequenceClassification
from transformers.optimization import get_linear_scheduler_with_warmup
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def main():
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()
    
    config_path = Path(args.config_file)
    config = load_config(config_path)

    assert "lm_model_name" in config

    DEVICE = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        DEVICE = "cpu"
    
    OUTPUT_DIR = Path(f'../models/{config["lm_model_name"]}_finetuned')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with timer("Loading and preparing data"):
        df_train, df_test, df_val = load_data(
            train_file, test_file, test_labels_file, 
            train_size=config.get("train_size", 1.0), 
            validation=config.get("validation", False)
        )
        print(f"Train size: {len(df_train)}, Test size: {len(df_test)}, Val size: {len(df_val) if df_val is not None else 'N/A'}")

        train_dataset = prepare_dataset(df_train, text_column='comment_text', label_columns=toxicity_labels)
        test_dataset = prepare_dataset(df_test, text_column='comment_text', label_columns=toxicity_labels)
        if df_val is not None:
            val_dataset = prepare_dataset(df_val, text_column='comment_text', label_columns=toxicity_labels)
        else:
            val_dataset = None
        
        train_loader = DataLoader(train_dataset,batch_size=config.get("batch_size", 16), shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=config.get("batch_size", 16), shuffle=False)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset,batch_size=config.get("batch_size", 16), shuffle=False)
        else:
            val_loader = None

    
    with timer("Training model"):
        model = BertForSequenceClassification.from_pretrained(
            config["lm_model_name"],
            num_labels=len(toxicity_labels)
        )
        model.to(DEVICE)
        optimizer = AdamW(
            model.parameters(),
            lr=config.get("lr", 2e-5),
            weight_decay=config.get("weight_decay", 0.01)
        )
        scheduler = get_linear_scheduler_with_warmup(
            optimizer,
            num_warmup_steps=config.get("warmup", 0)*len(train_loader)*config.get("epochs",2),
            num_training_steps=len(train_loader)*config.get("epochs",2)
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        model.train()
        for epoch in range(config.get("epochs", 2)):
            epoch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.get("epochs", 2)}')
            for batch in epoch_pbar:
                input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels.float())
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_pbar.set_postfix(loss=loss.item())
        model.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {OUTPUT_DIR}")

    with timer("Evaluating model"):
        model.eval()
        raw_preds, true_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                raw_preds.extend(outputs.logits.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        raw_preds = np.vstack(raw_preds)
        true_labels = np.vstack(true_labels)
        np.savetxt(OUTPUT_DIR/'true_labels.csv', true_labels, delimiter=',', fmt='%d')
        np.savetxt(OUTPUT_DIR/'predictions.csv', preds, delimiter=',', fmt='%d')

        roc_auc_total = 0
        for i, label in enumerate(toxicity_labels):
            roc_auc = roc_auc_score(true_labels[:, i], raw_preds[:, i])
            roc_auc_total += roc_auc
            print(f"ROC-AUC for {label}: {roc_auc:.4f}")
        roc_auc_avg = roc_auc_total / len(toxicity_labels)
        print(f"Avg ROC-AUC: {roc_auc_avg:.4f}")

        preds = (raw_preds > 0.5).astype(int)
        f1_total, acc_total, prec_total, rec_total = 0, 0, 0, 0
        for i, label in enumerate(toxicity_labels):
            tp = np.sum((true_labels[:, i] == 1) & (preds[:, i] == 1))
            tn = np.sum((true_labels[:, i] == 0) & (preds[:, i] == 0))
            fp = np.sum((true_labels[:, i] == 0) & (preds[:, i] == 1))
            fn = np.sum((true_labels[:, i] == 1) & (preds[:, i] == 0))
            acc = (tp + tn) / len(true_labels)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            print(f"{label}: Acc={acc:.3f}, F1={f1:.3f}, Rec={rec:.3f}, Prec={prec:.3f}")
            f1_total += f1
            acc_total += acc
            prec_total += prec
            rec_total += rec
        print(f"Avg F1: {f1_total/len(toxicity_labels):.3f}, Avg Acc: {acc_total/len(toxicity_labels):.3f}")

        submission = pd.DataFrame(df_test['id'],columns=['id'])
        for i, label in enumerate(toxicity_labels):
            submission[label] = raw_preds[:, i]
        submission.to_csv(OUTPUT_DIR/"submission.csv", index=False)

        
if __name__ == '__main__':
    main()