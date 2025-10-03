from config.base import toxicity_labels, train_file, test_file, test_labels_file
from src import load_data, prepare_dataset, load_config
from src.utils import timer
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path
import torch

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

    
    
