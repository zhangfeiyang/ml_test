#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
N = 500
import random
import ROOT
from ROOT import TH1F, TTree, TFile

f = TFile('data3.root',"read")
t = f.Get("t")

entries = t.GetEntries()
train_data = []
train_labels = []
for i in range(0,entries-100):
    t.GetEntry(i)
    a = t.GetLeaf('eplusm2').GetValue(0)
    b = t.GetLeaf('eeplusm2').GetValue(0)
    b = b/a
    em0 = t.GetLeaf('egm0').GetValue(0)
    em1 = t.GetLeaf('egm1').GetValue(0)
    em2 = t.GetLeaf('egm2').GetValue(0)
    em3 = t.GetLeaf('egm3').GetValue(0)
    em4 = t.GetLeaf('egm4').GetValue(0)
    em5 = t.GetLeaf('egm5').GetValue(0)
    em6 = t.GetLeaf('egm6').GetValue(0)
    em7 = t.GetLeaf('egm7').GetValue(0)

    m0 = t.GetLeaf('gm0').GetValue(0)
    m1 = t.GetLeaf('gm1').GetValue(0)
    m2 = t.GetLeaf('gm2').GetValue(0)
    m3 = t.GetLeaf('gm3').GetValue(0)
    m4 = t.GetLeaf('gm4').GetValue(0)
    m5 = t.GetLeaf('gm5').GetValue(0)
    m6 = t.GetLeaf('gm6').GetValue(0)
    m7 = t.GetLeaf('gm7').GetValue(0)

    train_data.append([[em0/m0,em1/m1,em2/m2,em3/m3,em4/m4,em5/m5,em6/m6,em7/m7],[m0,m1,m2,m3,m4,m5,m6,m7]])
    #train_data.append([em0/m0,em1/m1,em2/m2,em3/m3,em4/m4,em5/m5,em6/m6,em7/m7,m0,m1,m2,m3,m4,m5,m6,m7])
    #train_labels.append([b,b])
    train_labels.append([[a],[b]])

test_data = []
test_labels = []
for i in range(entries-100,entries):
    t.GetEntry(i)
#    a = t.GetLeaf('a').GetValue(0)
#    b = t.GetLeaf('b').GetValue(0)
    a = t.GetLeaf('eplusm2').GetValue(0)
    b = t.GetLeaf('eeplusm2').GetValue(0)
    b = b/a
    em0 = t.GetLeaf('egm0').GetValue(0)
    em1 = t.GetLeaf('egm1').GetValue(0)
    em2 = t.GetLeaf('egm2').GetValue(0)
    em3 = t.GetLeaf('egm3').GetValue(0)
    em4 = t.GetLeaf('egm4').GetValue(0)
    em5 = t.GetLeaf('egm5').GetValue(0)
    em6 = t.GetLeaf('egm6').GetValue(0)
    em7 = t.GetLeaf('egm7').GetValue(0)

    m0 = t.GetLeaf('gm0').GetValue(0)
    m1 = t.GetLeaf('gm1').GetValue(0)
    m2 = t.GetLeaf('gm2').GetValue(0)
    m3 = t.GetLeaf('gm3').GetValue(0)
    m4 = t.GetLeaf('gm4').GetValue(0)
    m5 = t.GetLeaf('gm5').GetValue(0)
    m6 = t.GetLeaf('gm6').GetValue(0)
    m7 = t.GetLeaf('gm7').GetValue(0)

    test_data.append([[em0/m0,em1/m1,em2/m2,em3/m3,em4/m4,em5/m5,em6/m6,em7/m7],[m0,m1,m2,m3,m4,m5,m6,m7]])
    #test_data.append([em0/m0,em1/m1,em2/m2,em3/m3,em4/m4,em5/m5,em6/m6,em7/m7,m0,m1,m2,m3,m4,m5,m6,m7])
    #test_labels.append([b,b])
    test_labels.append([[a],[b]])

import pandas as pd
column_names = [['e0','e1','e2','e3','e4','e5','e6','e7'],['e0','e1','e2','e3','e4','e5','e6','e7']]
#column_names = ['e0','e1','e2','e3','e4','e5','e6','e7','m0','m1','m2','m3','m4','m5','m6','m7']
train_data = [np.array(train_data[0]),np.array(train_data[1])]
test_data = [np.array(test_data[0]),np.array(test_data[1])]
train_labels = [np.array(train_labels[0]),np.array(train_labels[1])]
test_labels = [np.array(test_labels[0]),np.array(test_labels[1])]

#
#df = pd.DataFrame(train_data, columns=column_names)
#df.head(2)
#
##mean = train_data.mean(axis=0)
##std = train_data.std(axis=0)
##train_data = (train_data - mean) / std
##test_data = (test_data - mean) / std
##
##mean = train_labels.mean(axis=0)
##std = train_labels.std(axis=0)
##train_labels = (train_labels - mean) / std
##test_labels = (test_labels - mean) / std
#

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_model():
  #model = keras.Sequential([
  #  keras.layers.Dense(64, activation=tf.nn.relu, 
  #      input_shape=(train_data.shape[1],)),
  #  keras.layers.Dense(64, activation=tf.nn.relu),
  #  keras.layers.Dense(1)
  #  #keras.layers.Dense(2)
  #])

  a1 = Input(shape=(8,))
  a2 = Input(shape=(8,))
  b1 = Dense(1)(a1)
  b2 = Dense(1)(a2)

  optimizer = tf.train.RMSPropOptimizer(0.001)
  model = Model(inputs=[a1,a2],outputs=[b1,b2])

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch.

EPOCHS = 2

# Store training stats
#a0 = keras.layers.Input(shape=(32,))
#a1 = keras.layers.Input(shape=(32,))
#a2 = keras.layers.Input(shape=(32,))
#a3 = keras.layers.Input(shape=(32,))
#a4 = keras.layers.Input(shape=(32,))
#a5 = keras.layers.Input(shape=(32,))
#a6 = keras.layers.Input(shape=(32,))
#a7 = keras.layers.Input(shape=(32,))
#a8 = keras.layers.Input(shape=(32,))
#b1 = keras.layers.Dense(32)(a1)
#b2 = keras.layers.Dense(32)(a1)
#
#model = keras.models.Model(inputs=[a0,a1,a2,a3,a4,a5,a6,a7,a8],outputs=[b1,b2])
#
#model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
#model.add(Activation('softmax'))
#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer='sgd')
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
  plt.ylim([0,0.01])
  plt.show()

history = model.fit(train_data, train_labels, epochs=EPOCHS,
        validation_split=0.5, verbose=2)

#plot_history(history)
#
##predict_x = np.array([[10,20,70],[10,23,65]])
#predict_y = model.predict(x=test_data, batch_size=None, verbose=0, steps=None)
#print("predict_y is ")
#print(predict_y)
#print("true is ")
#print(test_labels)
