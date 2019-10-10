#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:22:29 2019

@author: vikram
"""
import numpy as np
from PIL import Image
import pickle
digit_detect_pkl = open('static/model/digit_predict.pkl', 'rb')
model = pickle.load(digit_detect_pkl)
print(model)
image = Image.open("final_image.jpg")
image = np.asarray(image)
image = image.reshape(1,28,28,1)
r = model.predict_classes(image)
print('predicted hand write digits is>>',r[0])




#import tensorflow as tf

from keras.models import load_model
model = load_model('static/model/model.h5')
print(model)

import numpy as np
from PIL import Image
image = Image.open("final_image.jpg")
image = np.asarray(image)
image = np.expand_dims(image, axis = 0)
image = image.reshape(1,28,28,1)
r = model.predict(image)
print('predicted hand writte digits is>>',r[0])

import matplotlib.pyplot as plt
index = [0,1,2,3,4,5,6,7,8,9]
plt.bar(index, r[0])
plt.xlabel('DIGITS', fontsize=15)
plt.ylabel('PREDICTED', fontsize=15)
#plt.xticks(index, label, fontsize=10, rotation=45)
plt.title('WRITTE DIGITS')
plt.show()
 

"""
he error message TypeError: Cannot interpret feed_dict key as Tensor: Tensor Tensor("...", dtype=dtype) 
is not an element of this graph can also arise
 in case you run a session outside of the scope of its with statement. Consider:
"""
#solve this error use it-->

from keras import backend as K
#Before prediction
K.clear_session()

#After prediction
K.clear_session()






