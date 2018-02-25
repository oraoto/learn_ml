#%% Import
import numpy as np
import pandas as pd
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import math
import sys

#%%
def read_file(path):
    with open(path, encoding="utf-8") as file:
        data = [line.split("\t") for line in file.readlines()]
        return pd.DataFrame(data, columns=["category", "text"])

#%%
path = "./text-classification/cnews/cnews.train.txt"
train_data = read_file(path)

#%%
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
encoder.fit(train_data)

#%%
train_data, train_labels = encoder.transform(train_data)

#%%
def mlp(batch_size, vocab_size, text_len, classes, features=64, train=True):
    text = nn.Variable([batch_size, text_len])

    with nn.parameter_scope("text_embed"):
        embed = PF.embed(text, n_inputs=vocab_size, n_features=features)
    print("embed", embed.shape)

    embed = F.reshape(embed, (batch_size, 1, text_len, features))
    print("embed", embed.shape)

    with nn.parameter_scope("conv1"):
        conv1 = PF.convolution(embed, 256, kernel=(5, features), stride=(1, 1))
        conv1 = PF.batch_normalization(conv1, batch_stat=train)
        conv1 = F.relu(conv1)
    print("conv1", conv1.shape)

    with nn.parameter_scope("max_pool1"):
        pool1 = F.max_pooling(conv1, kernel=(conv1.shape[2], 1))
    print("pool1", pool1.shape)

    with nn.parameter_scope("fc1"):
        fc1 = PF.affine(pool1, 32)
        fc1 = PF.batch_normalization(fc1, batch_stat=train)
        fc1 = F.relu(fc1)
    print("fc1", fc1.shape)

    with nn.parameter_scope("output"):
        y = PF.affine(fc1, classes)

    t = nn.Variable([batch_size, 1])

    _loss = F.softmax_cross_entropy(y, t)
    loss = F.reduce_mean(_loss)

    return text, y, loss, t

#%% Train
nn.clear_parameters()
batch_size = 1024
epochs = 10
learning_rate = 0.025

X, y, loss, t = mlp(batch_size, len(encoder.wtoi), encoder.text_len, len(np.unique(train_labels)))
solver = S.Adam(learning_rate)
solver.set_parameters(nn.get_parameters())
nn.save_parameters("text-cnn.h5")

#%%
iterations = math.ceil(len(train_data) / batch_size)
print(iterations)
all_loss = []
all_average_loss = []
min_loss =0.5

for epoch in range(epochs):
    total_loss = 0

    index = np.random.permutation(len(train_data))
    train_data = train_data[index]
    train_labels = train_labels[index]

    for i in range(iterations):
        d = train_data[i * batch_size : (i + 1) * batch_size]
        l = train_labels[i * batch_size : (i + 1) * batch_size]
        if len(d) < batch_size: # TODO
            continue
        X.d = d
        t.d = l.reshape((batch_size, 1))
        loss.forward()
        solver.zero_grad()
        loss.backward()
        solver.weight_decay(1e-5)
        solver.update()
        print("Iteration loss", epoch, i, loss.d.max())
        all_loss.append(loss.d.max())
        total_loss += loss.d.max()
    average_loss = total_loss / iterations
    print("Epoch loss", epoch, average_loss)
    if average_loss < min_loss:
        nn.save_parameters("./cnn-params/epoch_" + str(epoch) + ".h5")
        min_loss = average_loss
    all_average_loss.append(average_loss)

#%% Plot
fig = plt.gcf()
fig.set_size_inches(10, 5, forward=True)
plt.plot(all_loss, 'k-')
plt.title('Cross Entropy Loss per Batch')
plt.xlabel('Batch')
plt.yscale('log')
plt.ylabel('Cross Entropy Loss')
plt.show()

fig = plt.gcf()
fig.set_size_inches(10, 5, forward=True)
plt.plot(all_average_loss, 'k-')
plt.title('Average Cross Entropy Loss per Epoch')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Average Cross Entropy Loss')
plt.show()

#%% Validate
val_path = "./text-classification/cnews/cnews.val.txt"
val_data = read_file(val_path)
val_input, val_label = encoder.transform(val_data)

val_iterations = math.ceil(len(val_data) / batch_size)
predicts = []
truths = []
nn.load_parameters("cnn-params/epoch_4.h5")
X, y, loss, t = mlp(batch_size, len(encoder.wtoi), encoder.text_len, len(np.unique(train_labels)), train=False)

for i in range(val_iterations):
    
    d = val_input[i * batch_size : (i + 1) * batch_size] 
    l = val_label[i * batch_size : (i + 1) * batch_size]
    len_d = len(d)
    
    if len_d < batch_size:
        d = np.resize(d, (batch_size, encoder.text_len))

    X.d = d
    y.forward()
    predict = y.d.argmax(axis=1)

    if len_d < batch_size:
        predict = predict[:len_d]
    predicts.append(predict)
    truths.append(l)

truths  = np.concatenate(truths)
predicts = np.concatenate(predicts)

print(classification_report(truths, predicts))
print(confusion_matrix(truths, predicts))
