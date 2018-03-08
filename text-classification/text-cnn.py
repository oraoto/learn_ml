#%% Import
import numpy as np
import pandas as pd
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
#import matplotlib.pyplot as plt
import os
import math
import sys
import pickle

from nnabla.contrib.context import extension_context
cuda_device_id = 0
ctx = extension_context('cuda.cudnn', device_id=cuda_device_id)
nn.set_default_context(ctx)

#%% Config
text_len = 600
batch_size = 256
epochs = 20
learning_rate = 0.001

#%% Read data
def read_file(path):
    with open(path, encoding="utf-8") as file:
        data = [line.split("\t") for line in file.readlines()]
        return pd.DataFrame(data, columns=["category", "text"])

train_data = read_file("./data/cnews.train.txt")
validate_data = read_file("./data/cnews.val.txt")
test_data = read_file("./data/cnews.test.txt")

#%% Transform words and categories to integer
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

encoder = Encoder()
train_data = train_data.sample(frac=1)
encoder.fit(train_data)
train_data, train_labels = encoder.transform(train_data)
validate_data, validate_labels = encoder.transform(validate_data)
test_data, test_labels = encoder.transform(test_data)

#%% Define the network
def cnn(batch_size, vocab_size, text_len, classes, features=128, train=True):
    text = nn.Variable([batch_size, text_len])

    with nn.parameter_scope("text_embed"):
        embed = PF.embed(text, n_inputs=vocab_size, n_features=features)
    print("embed", embed.shape)

    embed = F.reshape(embed, (batch_size, 1, text_len, features))
    print("embed", embed.shape)

    combined = None
    for n in range(2, 6): # 2 - 5 gram
        with nn.parameter_scope(str(n) + "_gram"):
            with nn.parameter_scope("conv"):
                conv = PF.convolution(embed, 128, kernel=(n, features))
                conv = F.relu(conv)
            with nn.parameter_scope("pool"):
                pool = F.max_pooling(conv, kernel=(conv.shape[2], 1))
                if not combined:
                    combined = F.identity(pool)
                else:
                    combined = F.concatenate(combined, pool)

    if train:
        combined = F.dropout(combined, 0.5)

    with nn.parameter_scope("output"):
        y = PF.affine(combined, classes)

    t = nn.Variable([batch_size, 1])

    _loss = F.softmax_cross_entropy(y, t)
    loss = F.reduce_mean(_loss)

    return text, y, loss, t

def validate(data, labels):
    iterations = math.ceil(len(data) / batch_size)
    predicts = []
    truths   = []

    X, y, _, _ = cnn(batch_size, len(encoder.wtoi), encoder.text_len, len(np.unique(train_labels)), train=False)

    for i in range(iterations):
        d = data[i * batch_size : (i + 1) * batch_size]
        l = labels[i * batch_size : (i + 1) * batch_size]

        len_d = len(d)

        if len_d < batch_size:
            d = np.resize(d, (batch_size, text_len))

        X.d = d
        y.forward()
        predict = y.d.argmax(axis=1)

        if len_d < batch_size:
            predict = predict[:len_d]

        predicts.append(predict)
        truths.append(l)

    predicts = np.concatenate(predicts)
    truths = np.concatenate(truths)
    print(predicts)
    print(truths)

    print(classification_report(truths, predicts, digits=4))

    return predicts, truths

#% Train: solver
nn.clear_parameters()
cnn(batch_size, len(encoder.wtoi), encoder.text_len, len(np.unique(train_labels)))
solver = S.Adam(learning_rate)
solver.set_parameters(nn.get_parameters())
nn.save_parameters("text-cnn.h5")

#% Train: data collection
iterations = math.ceil(len(train_data) / batch_size)
all_loss = []
all_average_loss = []
min_loss = 10

#%% Train: loop
for epoch in range(epochs):
    X, y, loss, t = cnn(batch_size, len(encoder.wtoi), encoder.text_len, len(np.unique(train_labels)))
    total_loss = 0

    index = np.random.permutation(len(train_data))
    train_data = train_data[index]
    train_labels = train_labels[index]

    for i in range(iterations):
        d = train_data[i * batch_size : (i + 1) * batch_size]
        l = train_labels[i * batch_size : (i + 1) * batch_size]
        if len(d) < batch_size: # some data is skipped
            continue
        X.d = d
        t.d = l.reshape((batch_size, 1))
        loss.forward()
        solver.zero_grad()
        loss.backward()
        solver.weight_decay(1e-5)
        solver.update()
        all_loss.append(loss.d.max())
        total_loss += loss.d.max()

        if i % 20 == 0:
            print("Iteration loss", epoch, i, loss.d.max())

    average_loss = total_loss / iterations
    all_average_loss.append(average_loss)

    print("Epoch loss", epoch, average_loss)

    print("Validate")
    validate(validate_data, validate_labels)
    print("Train")
    validate(train_data, train_labels)

    if average_loss < min_loss:
        nn.save_parameters("./params_epoch_" + str(epoch) + ".h5")
        min_loss = average_loss

#%%
predict, truth = validate(validate_data, validate_labels)
predict, truth = validate(test_data, test_labels)
confusion_matrix(truth, predict)
