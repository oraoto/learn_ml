import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def read_file(path):
    with open(path, encoding="utf-8") as file:
        data = [line.split("\t") for line in file.readlines()]
        return pd.DataFrame(data, columns=["category", "text"])

class Encoder:
    def __init__(self, text_len=600):
        self.text_len = text_len
        self.wtoi = {}
        self.itow = {}
        self.label_encoder = LabelEncoder()

    def fit(self, X):
        for _, words in X['text'].items():
            words = list(str(words))
            for w in words:
                if w not in self.wtoi:
                    i = len(self.wtoi)
                    self.wtoi[w] = i
                    self.itow[i] = w
        self.label_encoder.fit(X['category'])

    def transform(self, X):
        text_dataset = []

        for _, words in X['text'].items():
            row = []
            for w in words:
                if w not in self.wtoi:
                    i = 0
                else:
                    i = self.wtoi[w]
                row.append(i)
            if len(row) > self.text_len:
                row[:self.text_len]
            text_dataset.append(np.resize(np.array(row), (self.text_len,))) # padding不是填0

        texts = np.stack(text_dataset)
        labels = self.label_encoder.transform(X['category'])
        return texts, labels

train_data = read_file("./data/cnews.train.txt")
validate_data = read_file("./data/cnews.val.txt")
test_data = read_file("./data/cnews.test.txt")

encoder = Encoder()

if os.path.exists("fast-text-encoder.pkl"):
    with open("fast-text-encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
else:
    encoder = Encoder()
    encoder.fit(train_data)
    with open("fast-text-encoder.pkl", "wb") as f:
        encoder = pickle.dump(encoder, f)

train_data, train_labels = encoder.transform(train_data)
validate_data, validate_labels = encoder.transform(validate_data)
test_data, test_labels = encoder.transform(test_data)
