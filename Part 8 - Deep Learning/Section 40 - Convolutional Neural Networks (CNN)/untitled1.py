#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:01:09 2018

@author: kay
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

#1. Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), 
                             activation='relu'))
#2. Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#3. Flattening
classifier.add(Flatten())

#4 Fully Connected layer
classifier.add(Dense(output_dim=120, activation='relu'))

classifier.add(Dense(output_dim=1, activation='sigmoid'))

#compiling
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics =['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

#fitting images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')
                    
classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=5,
    validation_data=test_set,
    validation_steps=2000)