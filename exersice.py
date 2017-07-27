#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:07:31 2017

@author: dandan
"""
from numpy import *
from keras.layers import Input, Dense
from keras.models import Model





def loaddataset(filename):
    numfeat=len(open(filename).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numfeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    
    return dataMat,labelMat

x,y=loaddataset("abalone.txt")

model=Sequential()
model.add(Dense(input_dim=8,output_dim=1,init="uniform",activation='linear'))

model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
model.fit(array(x),array(y),epochs=100,batch_size=32)
weights = model.layers[0].get_weights() 
w_final = weights[0][0][0] 
b_final = weights[1][0] 
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))
  

    