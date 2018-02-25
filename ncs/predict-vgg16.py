from mvnc import mvncapi as mvnc
import cv2
import numpy
import sys
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

img = load_img(sys.argv[1], target_size=(224, 224))
img = img_to_array(img)
img = preprocess_input(img)

# print("VGG16")

# vgg16 = VGG16()
# result = vgg16.predict(img.reshape(1, 224, 224, 3))
# print(decode_predictions(result))

print("NCS")

devices = mvnc.EnumerateDevices()

device = mvnc.Device(devices[0])
device.OpenDevice()

with open("graph", mode='rb') as f:
    graphfile = f.read()

graph = device.AllocateGraph(graphfile)

graph.LoadTensor(img.astype(numpy.float16), 'user object')
output, userobj = graph.GetResult()

graph.DeallocateGraph()
device.CloseDevice()

result = decode_predictions(output.reshape(1, 1000))
print(result)
