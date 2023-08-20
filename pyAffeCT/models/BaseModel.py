# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 rjfang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


class BaseModel(nn.Module):
    # Default hyperparameters
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.001
    num_epochs = 20
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, input_shape, num_classes):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    
    def forward(self, X):
        raise NotImplementedError()

    
    def init_weights(self, method):
        self.apply(self._init_layer_weight)
        
    def _init_layer_weight(self, layer):
        if type(layer) is nn.Linear:
            nn.init.normal_(layer.weight, std=0.001)
                
    
    def train(self, train_loader, test_loader=None):
        # transfer to the device
        self.to(self.device)
        
        loss_list = []
        accuracy_list = []
        
        # Iterate through all Epochs
        for epoch in range(self.num_epochs):
            # Iterate through training dataset
            for i, train_data in enumerate(train_loader, 0):
                # Flatten data and load data/labels onto GPU
                train_data, labels = train_data[0].to(self.device), train_data[1].to(self.device)
                _, labels = torch.max(labels,1)
    
                # Zero collected gradients at each step
                self.optimizer.zero_grad()
                # Forward Propagate
                outputs = self(train_data)
                # Calculate Loss
                loss = self.loss_fn(outputs, labels)
                # Back propagate
                loss.backward()
                # Update weights
                self.optimizer.step()
                
            print('Epoch [%d/%d],  Loss: %.4f'
                        %(epoch+1, self.num_epochs, loss.item()))
            loss_list.append(loss.item())
            if test_loader is not None:
                accuracy_list.append(self.test_accuracy(test_loader))
        return loss_list, accuracy_list
    
    def test_accuracy(self, test_loader):
        
        self.to(self.device)

        correct = 0
        total = 0
        with torch.no_grad():
            for test_data in test_loader:
                test_data, labels = test_data[0].cuda(), test_data[1].cuda()
                outputs = self(test_data)
                  
                """Calculate numeber of correct samples"""
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    outputs = outputs.argmax(axis=1)
                _, labels = torch.max(labels,1)
                cmp = outputs.type(labels.dtype) == labels
                correct += float(cmp.type(labels.dtype).sum())
                total += labels.numel()
            accuracy = correct/total
        print('Accuracy of the network on the test data: %.3f %%' % (100*accuracy))
    
        return accuracy