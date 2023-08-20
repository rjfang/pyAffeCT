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

class RNN(BaseModel):
    """
    This is an implementation of Recurrent Neural Network
    """

    
    def __init__(self, input_shape, num_classes, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, fc_dropout=0.):
        
        super().__init__(input_shape, num_classes)
        
        self.rnn = nn.RNN(input_shape[1], hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.fc = nn.Linear(hidden_size * (1 + bidirectional), num_classes)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def forward(self, x): 
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.fc(self.dropout(output))
        return output