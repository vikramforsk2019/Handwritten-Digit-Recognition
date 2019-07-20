#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:22:29 2019

@author: vikram
"""
import pickle
digit_detect_pkl = open('static/model/digit_predict.pkl', 'rb')
model = pickle.load(digit_detect_pkl)

print(model)




import tensorflow as tf

from keras.models import load_model

model = load_model('static/model/model.h5')

print(model)

import numpy as np
from PIL import Image
image = Image.open("Screenshot_2019-07-19 MNIST Handwritten text recognition using keras.png")
image = np.asarray(image)
image = np.expand_dims(image, axis = 0)
image.shape
r = model.predict_classes(image)


