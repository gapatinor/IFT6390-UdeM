#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np


# In[ ]:


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        self.dims = dims
        
        if self.seed is not None:
            np.random.seed(self.seed)
 
        self.weights = {}
        self.all_dims = [self.dims[0]] + list(self.hidden_dims) + [self.dims[1]]
        
        for layer_n in range(1, self.n_hidden + 2):
            val = 1/np.sqrt(self.all_dims[layer_n-1])
            weights_a = np.random.uniform(low=-val, high=val, size = (self.all_dims[layer_n-1], self.all_dims[layer_n], ))
            self.weights[f"W{layer_n}"] = weights_a
            self.weights[f"b{layer_n}"] = np.zeros((1, self.all_dims[layer_n]))

    def relu(self, x, grad):
        if grad:
            z_output = []
            for x_s in x:
                x_1 = np.sign(x_s)
                z_1 = np.maximum(x_1, 0)
                z_output.append(z_1)
            z = np.asarray(z_output)
            
        else:
            z = np.maximum(0, x)
        return z

    def sigmoid(self, x, grad=False):
        if grad:
            x_1 = 1/(1 + np.exp(-x))
            z = x_1 * (1 - x_1)
        else:
            z = 1/(1 + np.exp(-x))
        return z

    def tanh(self, x, grad=False):
        if grad:
            x_1 = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
            z = 1 - np.square(x_1)
        else:
            z = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        return z

    def activation(self, x, grad = False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")

    def softmax(self, x):
        if len(np.asarray(x).shape) != 1:
            z_output = []
            for x_s in x:
                x_1 = np.array(x_s) - np.max(x_s)
                y = np.exp(x_1)
                z = y/np.sum(y)
                z_output.append(z)
        else:
            x_1 = np.array(x) - np.max(x)
            y = np.exp(x_1)
            z_output = y/np.sum(y)
        return np.asarray(z_output)

    def forward(self, x):
        cache = {}
      
        Z_idx_1 = x
        for idx in range(0, len(self.all_dims)-1):  
            Z_idx = Z_idx_1
            
            W = self.weights[f"W{idx + 1}"] 
            b = self.weights[f"b{idx + 1}"] 

            A_idx_1 = np.dot(Z_idx, W) + b

            if idx != len(self.all_dims)-2:
                #before the last layer
                Z_idx_1 = []
                for A_idx_1_s in A_idx_1:
                    Z = self.activation(A_idx_1_s)
                    Z_idx_1.append(Z)
                Z_idx_1 = np.asarray(Z_idx_1)
            else:
                # Final layer with softmax activation
                Z_idx_1 = self.softmax(A_idx_1)

            cache[f"Z{idx}"] = Z_idx
            cache[f"A{idx + 1}"] = A_idx_1

        cache[f"Z{len(self.all_dims)-1}"] = Z_idx_1
        length_dict = [value.shape for value in cache.values()]
        return cache
    
    def backward(self, cache, labels):
        grads = {}
        output = np.asarray(cache[f"Z{self.n_hidden + 1}"])
        dA_idx = np.subtract(output, labels)
        
        Z_idx_1 = np.asarray(cache[f"Z{self.n_hidden}"])
        dW_idx = np.dot(Z_idx_1.T, dA_idx)/(self.batch_size)
        db_idx = np.sum(dA_idx, axis=0, keepdims=True)/self.batch_size
        
        W_idx = self.weights["W" + str(self.n_hidden + 1)]
        dZ_idx_1 = np.dot(dA_idx, W_idx.T) 
        
        grads[f"dW{self.n_hidden + 1}"] = dW_idx
        grads[f"db{self.n_hidden + 1}"] = db_idx
        grads[f"dA{self.n_hidden + 1}"] = dA_idx
        grads[f"dZ{self.n_hidden}"] = dZ_idx_1
        
        for idx in range(self.n_hidden, 0, -1):
            idx_1 = idx - 1
            
            A_idx = cache["A" + str(idx)]
            dA_idx = np.multiply(dZ_idx_1, self.activation(A_idx, grad = True))
            
            Z_idx_1 = cache["Z" + str(idx_1)]
            dW_idx = np.dot(Z_idx_1.T, dA_idx)/self.batch_size
            db_idx = np.sum(dA_idx, axis=0, keepdims=True)/self.batch_size

            W_idx = self.weights["W" + str(idx)]
            dZ_idx_1 = np.dot(dA_idx, W_idx.T)

            grads[f"dW{idx}"] = dW_idx
            grads[f"db{idx}"] = db_idx
            grads[f"dA{idx}"] = dA_idx
            if idx > 1:
                grads[f"dZ{idx_1}"] = dZ_idx_1
                
        return grads


    
    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - np.multiply(self.lr, grads[f"dW{layer}"])
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - np.multiply(self.lr, grads[f"db{layer}"]) 


    def one_hot(self, y):
        m = self.n_classes 
        mask = np.zeros(m)
        output = []
        for y_s in y:
            mask_y = np.copy(mask)
            mask_y[y_s] = mask_y[y_s] + 1
            output.append(mask_y)
        return np.asarray(output)

    def loss(self, prediction, labels):
        prediction = np.asarray(prediction)
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon

        N = prediction.shape[0]
        loss = -np.sum(labels*np.log(prediction))/N
        return loss

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy


# In[ ]:




