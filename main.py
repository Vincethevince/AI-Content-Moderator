from fastapi import FastAPI
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import uvicorn

app = FastAPI()
model = BertForSequenceClassification.from_pretrained('models/bert_finetuned_baseline')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

@app.post("/predict")
async def predict(data: dict):
    text = data.get("text")
    if not text:
        return {"detail": [{"type": "missing", "loc": ["body", "text"], "msg": "Field required", "input": None}]}
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.sigmoid(outputs.logits) > 0.5
        scores = torch.sigmoid(outputs.logits).cpu().numpy().tolist()
    outputs = [f"Your comment: '{text}' is\n"]
    for i, label in enumerate(label_cols):
        label_output = "not " * (1 if not preds[0][i].item() else 0) + f'{label} (Score: {scores[0][i]:.3f})\n'
        outputs.append(label_output)

    return {"response":"".join(outputs)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)