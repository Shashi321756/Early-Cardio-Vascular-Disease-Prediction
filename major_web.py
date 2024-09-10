# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:23:33 2024

@author: shash
"""

import numpy as np
from PIL import Image
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from google.colab.patches import cv2_imshow


# loading the saved model
loaded_model = pickle.load(open('D:/mini_project/major/trained_model.sav', 'rb'))
input_image_path = input('Path of the image to be predicted:')

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

input_image_resize = cv2.resize(input_image, (32,32))

input_image_scaled = input_image_resize/255

image_reshaped = np.reshape(input_image_scaled, [1,32,32,3])



input_prediction = model.predict(image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label == 0:
  print('The image represents a airplane')
elif input_pred_label == 1:
  print('The image represents a automobile')
elif input_pred_label == 2:
  print('The image represents a bird')
elif input_pred_label == 3:
  print('The image represents a cat')
elif input_pred_label == 4:
  print('The image represents a Deer')
elif input_pred_label == 5:
  print('The image represents a Dog')
elif input_pred_label == 6:
  print('The image represents a frog')
elif input_pred_label == 7:
  print('The image represents a Horse')
elif input_pred_label == 8:
  print('The image represents a ship')
elif input_pred_label == 9:
  print('The image represents a truck')