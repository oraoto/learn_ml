#%%
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
import tensorflow as tf

#%%
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#%%
cnn = models.Sequential()
cnn.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Conv2D(32, 3, activation='relu'))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Conv2D(64, 3, activation='relu'))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Flatten())
cnn.add(layers.Dense(256, activation='relu'))
cnn.add(layers.Dropout(0.5))
cnn.add(layers.Dense(10, activation='softmax'))
cnn.summary()

cnn.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

history = cnn.fit(x_train, y_train, epochs=10, batch_size=128)

print(cnn.evaluate(x_test, y_test))

#%%
with open("model.json", "w") as file:
    file.write(cnn.to_json())
cnn.save_weights("weights.h5")
