#!/usr/bin/python3
import tensorflow as tf
from tensorflow import keras
import numpy as np
N = 500
import random
import ROOT
from ROOT import TH1F, TTree, TFile
import os
from tensorflow.keras.models import load_model


energy = [6.13,2.5057,0.6617,1.022,1.4608,0.8345,4.945,2.223]

f = TFile('data3.root',"read")
t = f.Get("t")


#for i in range(0,1000):
#    t.GetEntry(i)
##    a = t.GetLeaf('a').GetValue(0)
##    b = t.GetLeaf('b').GetValue(0)
#    a = t.GetLeaf('eplusm2').GetValue(0)
#    b = t.GetLeaf('eeplusm2').GetValue(0)
#    b = b/a
#    em0 = t.GetLeaf('egm0').GetValue(0)
#    em1 = t.GetLeaf('egm1').GetValue(0)
#    em2 = t.GetLeaf('egm2').GetValue(0)
#    em3 = t.GetLeaf('egm3').GetValue(0)
#    em4 = t.GetLeaf('egm4').GetValue(0)
#    em5 = t.GetLeaf('egm5').GetValue(0)
#    em6 = t.GetLeaf('egm6').GetValue(0)
#    em7 = t.GetLeaf('egm7').GetValue(0)
#
#    m0 = t.GetLeaf('gm0').GetValue(0)
#    m1 = t.GetLeaf('gm1').GetValue(0)
#    m2 = t.GetLeaf('gm2').GetValue(0)
#    m3 = t.GetLeaf('gm3').GetValue(0)
#    m4 = t.GetLeaf('gm4').GetValue(0)
#    m5 = t.GetLeaf('gm5').GetValue(0)
#    m6 = t.GetLeaf('gm6').GetValue(0)
#    m7 = t.GetLeaf('gm7').GetValue(0)
#
#    test_data[0].append([em0/m0,em1/m1,em2/m2,em3/m3,em4/m4,em5/m5,em6/m6,em7/m7])
#
#    mean = [m0,m1,m2,m3,m4,m5,m6,m7]
#    for i in range(0,len(mean)):
#        mean[i] = mean[i]/energy[i]
#    for i in range(0,len(mean)):
#        mean[i] = mean[i]/mean[1]
#    test_data[1].append(mean)
#
#    #test_data.append([em0/m0,em1/m1,em2/m2,em3/m3,em4/m4,em5/m5,em6/m6,em7/m7,m0,m1,m2,m3,m4,m5,m6,m7])
#    #test_labels.append([b,b])
#    test_labels[0].append([b])
#    #test_labels[1].append([b])

test_data = [[],[]]
test_labels = [[]]

Datas0 = []
Datas1 = []
file = open('source','r')
for line in file:
    datas = line.split(' ')
    energy = float(datas[0])
    mean = float(datas[1])
    sigma = float(datas[3])
    Datas0.append(sigma/mean)
    Datas1.append(mean/energy/2990.13)

test_data[0].append(Datas0)
test_data[1].append(Datas1)
test_data = [np.array(test_data[0]),np.array(test_data[1])]

print(test_data)

model = load_model('model8.h5')
predict_y = model.predict(test_data, batch_size=None, verbose=2, steps=None)
print('predict_y is ',predict_y)
#h = TH1F('h','',100,0,0)
#for i in range(0,len(predict_y)):
#    print(predict_y[0])    
#    h.Fill(predict_y[i]/test_labels[0][i]-1)
#print('rms is',h.GetRMS(),' and mean is',h.GetMean())
