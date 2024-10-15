#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 04:18:20 2018

@author: sadievrenseker
"""
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D
import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS = 0

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd convolution layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection (Dense layers)
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image Data Generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size=(64, 64),
                                                 batch_size=1,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size=(64, 64),
                                            batch_size=1,
                                            class_mode='binary')

# Training the model
classifier.fit(training_set,
               steps_per_epoch=8000,  # equivalent to samples_per_epoch
               epochs=1,
               validation_data=test_set,
               validation_steps=2000)  # equivalent to nb_val_samples

# Predictions
import numpy as np
import pandas as pd

test_set.reset()
pred = classifier.predict(test_set, verbose=1)

# Apply threshold to predictions
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0

print('Prediction completed')

# Get test labels
test_labels = []
for i in range(len(test_set)):
    test_labels.extend(np.array(test_set[i][1]))

print('Test labels retrieved')

# Save the results
dosyaisimleri = test_set.filenames
sonuc = pd.DataFrame()
sonuc['dosyaisimleri'] = dosyaisimleri
sonuc['tahminler'] = pred
sonuc['test'] = test_labels

# Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, pred)
print(cm)
