#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
N = 500
import random
import ROOT
from ROOT import TH1F, TTree, TFile

f = TFile('data0_10000.root',"read")
t = f.Get("t")

entries = t.GetEntries()
train_data = []
train_labels = []
for i in range(0,entries-9):
    t.GetEntry(i)
    a = t.GetLeaf('a').GetValue(0)
    b = t.GetLeaf('b').GetValue(0)
    em0 = t.GetLeaf('em0').GetValue(0)
    em1 = t.GetLeaf('em1').GetValue(0)
    em2 = t.GetLeaf('em2').GetValue(0)
    em3 = t.GetLeaf('em3').GetValue(0)
    em4 = t.GetLeaf('em4').GetValue(0)
    em5 = t.GetLeaf('em5').GetValue(0)
    em6 = t.GetLeaf('em6').GetValue(0)
    em7 = t.GetLeaf('em7').GetValue(0)
    em8 = t.GetLeaf('em8').GetValue(0)

    m0 = t.GetLeaf('m0').GetValue(0)
    m1 = t.GetLeaf('m1').GetValue(0)
    m2 = t.GetLeaf('m2').GetValue(0)
    m3 = t.GetLeaf('m3').GetValue(0)
    m4 = t.GetLeaf('m4').GetValue(0)
    m5 = t.GetLeaf('m5').GetValue(0)
    m6 = t.GetLeaf('m6').GetValue(0)
    m7 = t.GetLeaf('m7').GetValue(0)
    m8 = t.GetLeaf('m8').GetValue(0)

    train_data.append([em0/m0,em1/m1,em2/m2,em3/m3,em4/m4,em5/m5,em6/m6,em7/m7,em8/m8])
    train_labels.append([a,b])

test_data = []
test_labels = []
for i in range(entries-9,entries):
    t.GetEntry(i)
    a = t.GetLeaf('a').GetValue(0)
    b = t.GetLeaf('b').GetValue(0)
    em0 = t.GetLeaf('em0').GetValue(0)
    em1 = t.GetLeaf('em1').GetValue(0)
    em2 = t.GetLeaf('em2').GetValue(0)
    em3 = t.GetLeaf('em3').GetValue(0)
    em4 = t.GetLeaf('em4').GetValue(0)
    em5 = t.GetLeaf('em5').GetValue(0)
    em6 = t.GetLeaf('em6').GetValue(0)
    em7 = t.GetLeaf('em7').GetValue(0)
    em8 = t.GetLeaf('em8').GetValue(0)

    m0 = t.GetLeaf('m0').GetValue(0)
    m1 = t.GetLeaf('m1').GetValue(0)
    m2 = t.GetLeaf('m2').GetValue(0)
    m3 = t.GetLeaf('m3').GetValue(0)
    m4 = t.GetLeaf('m4').GetValue(0)
    m5 = t.GetLeaf('m5').GetValue(0)
    m6 = t.GetLeaf('m6').GetValue(0)
    m7 = t.GetLeaf('m7').GetValue(0)
    m8 = t.GetLeaf('m8').GetValue(0)

    test_data.append([em0/m0,em1/m1,em2/m2,em3/m3,em4/m4,em5/m5,em6/m6,em7/m7,em8/m8])
    test_labels.append([a,b])

import pandas as pd
column_names = ['e0','e1','e2','e3','e4','e5','e6','e7','e8']
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)
#
df = pd.DataFrame(train_data, columns=column_names)
df.head(2)
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
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, 
        input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(2)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch.

EPOCHS = 500

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

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=2)

#predict_x = np.array([[10,20,70],[10,23,65]])
predict_y = model.predict(x=test_data, batch_size=None, verbose=0, steps=None)
print("predict_y is ")
print(predict_y)
print("true is ")
print(test_labels)
