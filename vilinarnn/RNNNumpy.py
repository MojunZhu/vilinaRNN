'''
Created on Feb 11, 2017

@author: mozhu
'''
import nltk
import numpy as np
from vilinarnn.utils import softmax
from nltk.metrics.aline import delta
from datetime import datetime
import sys

class RNNNumpy(object):
    '''
    classdocs
    '''

    def __init__(self, word_dim, hidden_dim = 100, bptt_truncate = 4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters 
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        
    def forward_propagation(self, x):
        # The total number of time steps 
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later
        # we add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T+1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are inding U by x[t]. This is the same as multiplying U with a one-hot vector
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o,s]
    
    def predit(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis = 1)
    
    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentences
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            print(y[i])
            print(len(y[i]))
            print(np.arange(len(y[i])))
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
    
    def calculate_loss(self, x, y):
        # divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y)/N
    
    def bptt(self, x, y):
        T = len(y)
        # perform forward propagation
        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t])*(1-(s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0,t-self.bptt_truncate), t+1)[::-1]:
                #print "Backpropagation step t=%d bptt step=%d" % (t,bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                #update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
    
    # leave place for gradient checking
    
    # Performs one step of SGD
    def numpy_sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # change parameters accroding to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
    
     # Outer SGD Loop
     # - model: The ENN model instance
     # - X-train: the training data set
     # - y_train: the training data labels
     # - learning_rate: Initial learning rate for SGD
     # - nepoch: Number of times of iterate through the complete dataset
     # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self, model, X_train, y_train, learning_rate = 0.005, nepoch = 100, evaluate_loss_after = 5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionlly evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_example_seen = %d epoch = %d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increate
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each traning example...
            for i in range(len(y_train)):
                # One SGD step
                model.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
                    

        
        
        
        
        
            