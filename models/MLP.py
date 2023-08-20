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

class MLP(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)
        
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape[-1]*input_shape[-2],500)
        self.fc2 = nn.Linear(500,500)
        self.fc3 = nn.Linear(500,500)
        self.fc4 = nn.Linear(500,num_classes)
        
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
    def forward(self, X):
        X = X.view(-1,self.input_shape[-1]*self.input_shape[-2])
        out1 = F.relu(self.fc1(X))
        out2 = F.relu(self.fc2(out1))
        out3 = F.relu(self.fc3(out2))
        out4 = self.fc4(out3)
        out = F.sigmoid(out4)
        return out
    
    
    
        