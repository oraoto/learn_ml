from mvnc import mvncapi as mvnc
import cv2
import numpy
import sys
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import time

# NCS
devices = mvnc.EnumerateDevices()

device = mvnc.Device(devices[0])
device.OpenDevice()

with open("graph", mode='rb') as f:
    graphfile = f.read()

graph = device.AllocateGraph(graphfile)

# OpenCV
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype('float32'))

    graph.LoadTensor(img.astype(numpy.float16), 'user object')

    output, userobj = graph.GetResult()

    result = decode_predictions(output.reshape(1, 1000))

    print(result[0])

    frame = cv2.putText(frame, str(result[0][0]), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

graph.DeallocateGraph()
device.CloseDevice()
