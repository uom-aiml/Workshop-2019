!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:48:32 2019
​
@author: antoinek
"""
​
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
​
# Import and Reshape the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape (60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)
​
# input image dimension
# Because we have 28 by 28, our model will have an input of dimenssion 28*28,1 => 784,1
img_rows, img_cols = 28, 28
​
batch_size =128;
input_shape = (img_rows, img_cols, 1)
​
​
# Change to 1 hot encoder
y_train = to_categorical(y_train)
#y_test_integer = y_test
y_test = to_categorical(y_test)
​
# 
inputs = Input(shape=input_shape)
flat = Flatten()(inputs)
hidden1 = Dense(64, activation='relu')(flat)
hidden2 = Dense(128, activation='relu')(hidden1)
output = Dense(10, activation='softmax')(hidden2)
model = Model(inputs=inputs, outputs=output)
​
print(model.summary())
plot_model(model,show_shapes= True, to_file='cnn.png')
​
opti= Adam(learning_rate=0.01)
model.compile(optimizer=opti,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs= 10, batch_size= batch_size, verbose=1, validation_data=(x_test, y_test)) 
​
