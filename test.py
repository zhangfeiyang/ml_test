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
    train_data.append([a,b,c])
    train_labels.append([sum])

test_data = []
test_labels = []
for i in range(0,N):
    a = random.gauss(10,1)
    b = random.gauss(20,5)
    c = random.gauss(50,15)
    sum = a+b+c
    test_data.append([a,b,c])
    test_labels.append([sum])
import pandas as pd
column_names = ['a','b','c']
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

df = pd.DataFrame(train_data, columns=column_names)
df.head(2)

#mean = train_data.mean(axis=0)
#std = train_data.std(axis=0)
#train_data = (train_data - mean) / std
#test_data = (test_data - mean) / std
#
#mean = train_labels.mean(axis=0)
#std = train_labels.std(axis=0)
#train_labels = (train_labels - mean) / std
#test_labels = (test_labels - mean) / std

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 2000

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0,5])
  plt.show()

plot_history(history)

de = ' '

while not de in 'q':
    de = input('enter q to exit')
    if 1<len(de):
        de = de[0]
test_predictions = model.predict(test_data).flatten()
#print(test_predictions)

for pre,data in zip(test_predictions,test_data):
    a = data[0]
    b = data[1]
    c = data[2]
    print(pre,a+b+c-pre,(a+b+c-pre)/(a+b+c)*100)
