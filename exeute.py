import sys
import os 
from Document_Classifier import *
from PIL import Image
import io


getClass = Document_Classifier()
im = Image.open("pancardsample2.jpg")
img_byte_arr = io.BytesIO()
im.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()
 
print(im)

print(getClass.imgClassification(img_byte_arr))
