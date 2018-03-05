#%%
import face_recognition
import sys
import os
from PIL import Image, ImageDraw

#%%
def find_face(path, crop):
    print(path)
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image, 2, model="hog")

    print(face_locations)

    pil_image = Image.fromarray(image)

    
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left) in face_locations:
        padding = (right - left) // 4
        if crop:
            box = (max(left - padding // 4, 0), max(top - padding * 2, 0), right + padding // 4, bottom + padding // 4)
            pil_image = pil_image.crop(box)
            break
        draw.rectangle((max(left - padding // 4, 0), max(top - padding * 2, 0), (right + padding // 4, bottom + padding // 4)), outline=(0, 0, 255))
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
    del draw

    return pil_image

if __name__ == '__main__':
    crop = len(sys.argv) > 2 and sys.argv[2] == "c"
    img = find_face(sys.argv[1], crop)

    img.show()
