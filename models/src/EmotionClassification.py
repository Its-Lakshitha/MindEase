import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast
import torch

# 1. Load Preprocessed Data

train_texts = pd.read_csv("data/processed/train_texts.csv", header=None)[0].tolist()
test_texts  = pd.read_csv("data/processed/test_texts.csv",  header=None)[0].tolist()

y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

print("Loaded preprocessed data:")
print(f"  Train texts: {len(train_texts)}")
print(f"  Test texts : {len(test_texts)}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape : {y_test.shape}")


# 2. Load DistilBERT Tokenizer

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

print("Tokenizer loaded.")


# 3. Tokenize Data

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128
)

print("Tokenization complete.")


# 4. PyTorch Dataset Class

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()  # float → for BCE loss
        return item

    def __len__(self):
        return len(self.labels)


# 5. Create Dataset Objects

train_dataset = EmotionDataset(train_encodings, y_train)
test_dataset  = EmotionDataset(test_encodings,  y_test)

print("PyTorch Dataset objects created:")
print(f"  Train dataset size: {len(train_dataset)}")
print(f"  Test dataset size : {len(test_dataset)}")
