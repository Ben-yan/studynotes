#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:37:57 2017

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
N=100
D=2
K=3
X=np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8')



class softmax_classifier:
    
    def __init__(self):
        
        self.N=N
        self.D=D
        self.K=K
        for j in range(K):
          ix = range(N*j,N*(j+1))
          r = np.linspace(0.0,1,N) # radius
          t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
          X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
          y[ix] = j
          
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()
       
    def train_eval(self,lr=1e-0,reg=1e-3,decay=0):
            #(learning_rate,regularization_strength,decay_rate):

#Normally we would want to preprocess the dataset so that each feature has zero mean 
#and unit standard deviation, but in this case the features are already in a nice range 
#from -1 to 1, so we skip this step.
#       initialization randomly   
        self.lr=lr
        self.reg=reg
        self.decay=decay
  # gradient descent loop
        W = 0.01 * np.random.randn(D,K)
        b = np.zeros((1,K))
        num_examples = X.shape[0]
        for i in range(200):
               
  # evaluate class scores, [N x K]
          scores = np.dot(X, W) + b 
  
  # compute the class probabilities
          exp_scores = np.exp(scores)
          probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
          corect_logprobs = -np.log(probs[range(num_examples),y])
          data_loss = np.sum(corect_logprobs)/num_examples
          reg_loss = 0.5*reg*np.sum(W*W)
          loss = data_loss + reg_loss
          if i % 10 == 0:
                   
              print ("iteration %d: loss %f" % (i, loss))
  
  # compute the gradient on scores
          dscores = probs
          dscores[range(num_examples),y] -= 1
          dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
          dW = np.dot(X.T, dscores)
          db = np.sum(dscores, axis=0, keepdims=True)
      
          dW += reg*W # regularization gradient
      
  # perform a parameter update
          W += -lr * dW
          b += -lr * db
            
        print("weights trained\n{0},\n bias trained{1}".format(W,b))
        score=np.dot(X,W)+b
        pre_class1=np.argmax(score,axis=1)
        self.score=score
        self.w=W
        self.pred=pre_class1
        self.b=b
        
        print("Accuracy with pure softmax model is {}".format(np.mean(pre_class1==y)))
        return self.w,self.b
# X - some data in 2dimensional np.array


def run_softmax():
    proc=softmax_classifier()
    w=proc.train_eval()[0]
    b=proc.train_eval()[1]

    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max,0.01),np.arange(y_min, y_max,0.01))
    im=np.c_[xx.ravel(), yy.ravel()]
   ##we need to know
    Z = np.dot(im,w)+b
    Z=np.argmax(Z,axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    
    plt.show()







def networknize():
    # initialize parameters randomly
    h = 100 # size of hidden layer
    W = 0.01 * np.random.randn(D,h)
    b = np.zeros((1,h))
    W2 = 0.01 * np.random.randn(h,K)
    b2 = np.zeros((1,K))
    
    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3 # regularization strength
    
    # gradient descent loop
    num_examples = X.shape[0]
    for i in range(10000):
      
      # evaluate class scores, [N x K]
      hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
      scores = np.dot(hidden_layer, W2) + b2
      
      # compute the class probabilities
      exp_scores = np.exp(scores)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
      
      # compute the loss: average cross-entropy loss and regularization
      corect_logprobs = -np.log(probs[range(num_examples),y])
      data_loss = np.sum(corect_logprobs)/num_examples
      reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
      loss = data_loss + reg_loss
      if i % 1000 == 0:
        print ("iteration %d: loss %f" % (i, loss))
      
      # compute the gradient on scores
      dscores = probs
      dscores[range(num_examples),y] -= 1
      dscores /= num_examples
      
      # backpropate the gradient to the parameters
      # first backprop into parameters W2 and b2
      dW2 = np.dot(hidden_layer.T, dscores)
      db2 = np.sum(dscores, axis=0, keepdims=True)
      # next backprop into hidden layer
      dhidden = np.dot(dscores, W2.T)
      # backprop the ReLU non-linearity
      dhidden[hidden_layer <= 0] = 0
      # finally into W,b
      dW = np.dot(X.T, dhidden)
      db = np.sum(dhidden, axis=0, keepdims=True)
      
      # add regularization gradient contribution
      dW2 += reg * W2
      dW += reg * W
      
      # perform a parameter update
      W += -step_size * dW
      b += -step_size * db
      W2 += -step_size * dW2
      b2 += -step_size * db2
    
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))
    
      
    
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max,0.01),
                         np.arange(y_min, y_max,0.01))
    q=np.c_[xx.ravel(), yy.ravel()]
    
    
    hidden_layer = np.maximum(0, np.dot(q, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    Z = predicted_class.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    
    plt.show()

def run_network():
    networknize()





