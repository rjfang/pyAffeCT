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

from pyAffeCT.models import BaseModel

class LSTM_FCN(BaseModel):
    """
    This is an implementation of Recurrent Neural Network
    """

    
    def __init__(self, input_shape, num_classes, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, fc_dropout=0.):
        
        super().__init__(input_shape, num_classes)
        
        self.lstm = nn.LSTM(input_shape[1], hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=128, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU()
        
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) + 128, num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def forward(self, x): 
        # LSTM
        lstm_output = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        lstm_output, _ = self.lstm(lstm_output) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        lstm_output = lstm_output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        
        
        # FCN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.gap(x)
        x = x.reshape([x.shape[0],x.shape[1]])
        
        # Concat

        x = torch.cat((lstm_output, x), dim=1)

        out = self.fc(x)
    
        return out
