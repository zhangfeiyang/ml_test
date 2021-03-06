#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
N = 500
import random

train_data = []
train_labels = []
for i in range(0,N):
    a = random.gauss(10,1)
    b = random.gauss(20,5)
    c = random.gauss(50,15)
    sum = a+b+c
    train_data.append([[a,b,c],[a,b,c]])
    train_labels.append([[sum],[sum]])

test_data = []
test_labels = []
for i in range(0,N):
    a = random.gauss(10,1)
    b = random.gauss(20,5)
    c = random.gauss(50,15)
    sum = a+b+c
    test_data.append([[a,b,c],[a,b,c]])
    test_labels.append([[sum],[sum]])
import pandas as pd
column_names = [['a','b','c'],['a','b','c']]
train_data = [np.array(train_data[0]),np.array(train_data[1])]
train_labels = [np.array(train_labels[0]),np.array(train_labels[1])]
test_data = [np.array(test_data[0]),np.array(test_data[1])]
test_labels = [np.array(test_labels[0]),np.array(test_labels[1])]

#df = pd.DataFrame(train_data, columns=column_names)
#df.head(2)

#mean = train_data.mean(axis=0)
#std = train_data.std(axis=0)
#train_data = (train_data - mean) / std
#test_data = (test_data - mean) / std
#
#mean = train_labels.mean(axis=0)
#std = train_labels.std(axis=0)
#train_labels = (train_labels - mean) / std
#test_labels = (test_labels - mean) / std

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_model():
  #model = keras.Sequential([
  #  keras.layers.Dense(64, activation=tf.nn.relu, 
  #                     input_shape=(train_data.shape[1],)),
  #  keras.layers.Dense(64, activation=tf.nn.relu),
  #  keras.layers.Dense(1)
  #])

  #optimizer = tf.train.RMSPropOptimizer(0.001)

  #model.compile(loss='mse',
  #              optimizer=optimizer,
  #              metrics=['mae'])
  
  a1 = Input(shape=(3,))
  a2 = Input(shape=(3,))
  b1 = Dense(1)(a1)
  b2 = Dense(1)(a1)
  b3 = Dense(1)(a1)

  optimizer = tf.train.RMSPropOptimizer(0.001)
  model = Model(inputs=[a1,a2],outputs=[b1,b2])
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])

  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch.

EPOCHS = 50
#
# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=2)
import matplotlib.pyplot as plt
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  #plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
  plt.plot(history.epoch, np.array(history.history['dense_mean_absolute_error']),
           label='Train Loss')
  #plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
  plt.plot(history.epoch, np.array(history.history['val_dense_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0,300])
  plt.show()

plot_history(history)

#predict_x = np.array([[10,20,70],[10,23,65]])
#predict_y = model.predict(x=predict_x, batch_size=None, verbose=0, steps=None)
#print("predict_y is ")
#print(predict_y)
