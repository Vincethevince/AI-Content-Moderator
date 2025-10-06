from pathlib import Path

toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

DATA_DIR = Path.cwd()/"data/jigsaw"
train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"
test_labels_file = DATA_DIR / "test_labels.csv"

MODEL_DIR = Path.cwd() / "models"
BERT_BASE_MODEL = MODEL_DIR / "bert-base-uncased"
