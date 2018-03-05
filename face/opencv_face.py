#%%
import face_recognition
import sys
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    scale = 4

    small_frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)

    face_locations = face_recognition.face_locations(small_frame, 1, model="hog")

    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left) in face_locations:
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale
        padding = (right - left) // 4
        draw.rectangle((max(left - padding // 4, 0), max(top - padding * 2, 0), (right + padding // 4, bottom + padding // 4)), outline=(0, 0, 255))
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
    del draw

    img = np.array(pil_image)
    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
