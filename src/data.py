import pandas as pd
from sklearn.model_selection import train_test_split
from .tokenize import MyTokenizer
import torch
from torch.utils.data import TensorDataset


def load_data(train_path, test_path, test_labels_path, train_size=1.0, validation=False):
    df_train = pd.read_csv(train_path)

    if train_size < 1.0:
        df_train = df_train.sample(frac=train_size, random_state=42).reset_index(drop=True)
    
    if validation:
        df_train, df_val =  train_test_split(df_train, test_size=0.1, random_state=42)
    else:
        df_val = None

    df_test = pd.read_csv(test_path)
    test_labels = pd.read_csv(test_labels_path)
    df_test = df_test.merge(test_labels, on='id')
    df_test = df_test[df_test["toxic"]!=-1].reset_index(drop=True)
    
    return df_train, df_test, df_val

def preprocess_data(df):
    pass

def prepare_dataset(df, text_column, label_columns):
    # Placeholder to add downsample_frac=0.0
    tokenizer = MyTokenizer(num_workers=4)
    label_vals = df[label_columns].values
    X = tokenizer.tokenize_batch(df[text_column].tolist())
    dataset = TensorDataset(
        torch.from_numpy(X['input_ids']),
        torch.from_numpy(X['attention_mask']),
        torch.tensor(label_vals, dtype=torch.int8)
    )

    return dataset