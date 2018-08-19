#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import ROOT
from ROOT import TH1F, TTree, TFile
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

f = TFile('fina.root',"read")
t = f.Get("evt")

entries = t.GetEntries()
train_data = [[],[]]
train_labels = []

for i in range(0,entries):
    print(i)
    PE_pmtID = []
    time_pmtID = []
    for k in range(0,1774):
        PE_pmtID.append(0)
        time_pmtID.append(1000000)
    t.GetEntry(i)
    PE = int(t.GetLeaf("totalPE").GetValue(0))
    t.GetEntry(i)
    edep = t.GetLeaf("edep").GetValue(0)
    for j in range(0,PE):
        pmtID = int(t.GetLeaf('pmtID').GetValue(j))
        hittime = t.GetLeaf('hitTime').GetValue(j)
        if pmtID<20000:
            PE_pmtID[int(pmtID/10)] = PE_pmtID[int(pmtID/10)]+1
            if hittime < time_pmtID[int(pmtID/10)]:
                time_pmtID[int(pmtID/10)] = hittime
    train_data[0].append(PE_pmtID)
    train_data[1].append(time_pmtID)
    if edep>0.999:
        train_labels.append([1])
    else:
        train_labels.append([0])

train_data = [np.array(train_data[0]),np.array(train_data[1])]
train_labels = np.array(train_labels)

def build_model():
  a1 = Input(shape=(1774,))
  a2 = Input(shape=(1774,))

  b1 = Dense(1774,activation='relu')(a1)
  b2 = Dense(1774,activation='relu')(a2)
  b2 = keras.layers.concatenate([b1, b2])


  #b2 = Dense(64,activation='relu')(b2)
  b2 = Dense(1,activation='relu')(b2)

  optimizer = tf.train.RMSPropOptimizer(0.001)
  optimizer = tf.keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
  model = Model(inputs=[a1,a2],outputs=[b2])
  model.compile(loss='mape',
                #optimizer='rmsprop',
                optimizer=optimizer,
                #loss_weights=[1., 0.2],metrics=['mape', 'mape'])
                metrics=['mape'])
  return model

model = build_model()
model.summary()

EPOCHS = 20
import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0,5])
  plt.show()

history = model.fit(train_data, train_labels, epochs=EPOCHS,
        validation_split=0.2, verbose=2, batch_size=20)

plot_history(history)
