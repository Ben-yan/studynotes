#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:04:51 2017

@author: Ben 
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import h5py as h


trainx=np.linspace(-1,1,100)
trainy=trainx*4+np.random.randn(*trainx.shape)*0.33
                               
model=Sequential()
model.add(Dense(input_dim=1,output_dim=1,init='uniform',activation='linear'))
'''
weights = model.layers[0].get_weights() 
w_init = weights[0][0][0] 
b_init = weights[1][0] 
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) ## Linear regression model is initialized with weight w: -0.03, b: 0.00
'''     
     
     
model.compile(optimizer='sgd',loss='mse')
model.fit(trainx,trainy,epochs=200)
weights = model.layers[0].get_weights() 
w_final = weights[0][0][0] 
b_final = weights[1][0] 
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))







                               
                               

