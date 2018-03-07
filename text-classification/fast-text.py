#%% Import
import numpy as np
import pandas as pd
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from encoder import encoder, train_data, train_labels, validate_data, validate_labels, read_file
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import math
import sys
import os

#%%
batch_size = 1024
epochs = 20
learning_rate = 0.01
param_file = "fast-text.h5"
features = 32

text_len = encoder.text_len
classes = len(np.unique(train_labels))
vocab_size = len(encoder.wtoi)

#%% Define the network
def fast_text(batch_size, vocab_size, text_len, classes, features, train=True):
    text = nn.Variable([batch_size, text_len])

    with nn.parameter_scope("text_embed"):
        embed = PF.embed(text, n_inputs=vocab_size, n_features=features)

    avg = F.mean(embed, axis=1)

    with nn.parameter_scope("output"):
        y = PF.affine(avg, classes)

    t = nn.Variable([batch_size, 1])

    _loss = F.softmax_cross_entropy(y, t)
    loss = F.reduce_mean(_loss)

    return text, y, loss, t

def validate(data, labels):
    iterations = math.ceil(len(data) / batch_size)
    predicts = []
    truths   = []

    X, y, _, _ = net(train=False)

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

    print(classification_report(truths, predicts))

    return predicts, truths


def net(train=True):
    return fast_text(batch_size, vocab_size, text_len, classes, features, train=True)

#% Train: solver
nn.clear_parameters()
net(train=True)
solver = S.Adam(learning_rate)
solver.set_parameters(nn.get_parameters())
nn.save_parameters(param_file)

#% Train: data collection
iterations = math.ceil(len(train_data) / batch_size)
all_loss = []
all_average_loss = []
min_loss = 10

#%% Train: loop
for epoch in range(epochs):
    X, y, loss, t = net(train=True)
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

        if i % 10 == 0:
            print("Iteration loss", epoch, i, i / iterations, loss.d.max())

    average_loss = total_loss / iterations
    all_average_loss.append(average_loss)

    print("Epoch loss", epoch, average_loss)
    print("Train:")
    validate(train_data, train_labels)
    print("Validate:")
    validate(validate_data, validate_labels)

    nn.save_parameters("./params_epoch_" + str(epoch) + ".h5")
    if average_loss < min_loss:
        nn.save_parameters(param_file)
        min_loss = average_loss

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
predict, truth = validate(validate_data, validate_labels)
print(confusion_matrix(truth, predict))
predict, truth = validate(train_data, train_labels)
print(confusion_matrix(truth, predict))

#              precision    recall  f1-score   support

#           0       0.99      0.99      0.99       500
#           1       0.97      0.88      0.92       500
#           2       0.97      0.72      0.83       500
#           3       0.81      0.93      0.86       500
#           4       0.93      0.89      0.91       500
#           5       0.94      0.98      0.96       500
#           6       0.92      0.94      0.93       500
#           7       0.92      0.97      0.94       500
#           8       0.92      0.99      0.95       500
#           9       0.95      0.97      0.96       500

# avg / total       0.93      0.93      0.93      5000

# [[496   0   0   0   4   0   0   0   0   0]
#  [  4 442   1   0   4  10   0  18  18   3]
#  [  1   5 362  81   6  14  11   1  11   8]
#  [  0   2   8 465   6   2   9   2   0   6]
#  [  0   1   2   2 446   0  17  21   8   3]
#  [  1   4   0   0   2 492   0   0   1   0]
#  [  0   3   0  14   5   0 472   1   2   3]
#  [  0   1   1   1   3   3   0 486   5   0]
#  [  0   0   1   0   1   1   0   2 494   1]
#  [  0   0   0  14   1   0   2   0   0 483]]

#              precision    recall  f1-score   support

#           0       1.00      0.99      1.00      5000
#           1       0.98      0.99      0.99      5000
#           2       0.98      0.97      0.98      5000
#           3       0.94      0.96      0.95      5000
#           4       0.95      0.96      0.96      5000
#           5       0.98      0.99      0.98      5000
#           6       0.96      0.96      0.96      5000
#           7       0.99      0.98      0.99      5000
#           8       0.96      0.97      0.96      5000
#           9       0.97      0.94      0.95      5000

# avg / total       0.97      0.97      0.97     50000

# [[4973    3    1    2    8    2    1    3    4    3]
#  [   2 4947    3    3    6   21    6    8    2    2]
#  [   0   14 4868   66   15   23    7    2    3    2]
#  [   0    5   29 4806   16    4   61    1    7   71]
#  [   4   10    6   15 4809   20   20   10   86   20]
#  [   2   20   11    7   16 4926    4    5    7    2]
#  [   1    5    3   77   29    0 4820    1   41   23]
#  [   4   15    3    6   19    9    4 4924   14    2]
#  [   3   11    2   16   62    5   42   14 4832   13]
#  [   2    7   17  136   62   10   48    1   24 4693]]
