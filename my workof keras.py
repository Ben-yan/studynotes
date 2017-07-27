#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:32:00 2017

@author: dandan
"""
import pydot
import pydot_ng
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils        
from keras.utils.vis_utils import plot_model  
import numpy as np

#load data
def load(filename):
    xmat=[];labelmat=[]
    for line in open(filename).readlines():
        currline=line.strip().split('\t')
        linearr=[]
        for i in range(21):
            linearr.append(currline[i])
        xmat.append(linearr)
        labelmat.append(currline[21])
    return xmat,labelmat

x,y=load("horseColicTraining.txt")
testx,testy=load('horseColicTest.txt')

#model
model=Sequential()
model.add(Dense(input_dim=21,output_dim=1,activation='relu',init='uniform'))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
model.fit(np.array(x),np.array(y),epochs=100,batch_size=32)

weight=model.layers[0].get_weights()[0]
print("weights worked out are \n {}".format(weight))

score=model.evaluate(np.array(testx),np.array(testy),batch_size=32)
print("\n test {0[0]} is {1[0]}, {0[1]} is {1[1]}".format(model.metrics_names,score))
plot_model(model,to_file='model.png',show_shapes=True)







    

