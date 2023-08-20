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

class ResNet(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)
        
        n_feature_maps = 64

        self.relu = nn.ReLU()
         
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=n_feature_maps, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(n_feature_maps)
         
        self.conv2 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(n_feature_maps)
         
        self.conv3 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(n_feature_maps)
         
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[1], out_channels=n_feature_maps, kernel_size=1, padding='same'),
            nn.BatchNorm1d(n_feature_maps)
        )
         
        self.conv4 = nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps*2, kernel_size=8, padding='same')
        self.bn4 = nn.BatchNorm1d(n_feature_maps*2)
         
        self.conv5 = nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=5, padding='same')
        self.bn5 = nn.BatchNorm1d(n_feature_maps*2)
         
        self.conv6 = nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm1d(n_feature_maps*2)
         
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(in_channels=n_feature_maps, out_channels=n_feature_maps*2, kernel_size=1, padding='same'),
            nn.BatchNorm1d(n_feature_maps*2)
        )
         
        self.conv7 = nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=8, padding='same')
        self.bn7 = nn.BatchNorm1d(n_feature_maps*2)
         
        self.conv8 = nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=5, padding='same')
        self.bn8 = nn.BatchNorm1d(n_feature_maps*2)
         
        self.conv9 = nn.Conv1d(in_channels=n_feature_maps*2, out_channels=n_feature_maps*2, kernel_size=3, padding='same')
        self.bn9 = nn.BatchNorm1d(n_feature_maps*2)
         
        self.shortcut3 = nn.BatchNorm1d(n_feature_maps*2)
         
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
         
        self.fc = nn.Linear(n_feature_maps*2, num_classes)
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
         
    def forward(self, x):
        # BLOCK 1
        conv_x = self.conv1(x)
        conv_x = self.bn1(conv_x)
        conv_x = self.relu(conv_x)
         
        conv_y = self.conv2(conv_x)
        conv_y = self.bn2(conv_y)
        conv_y = self.relu(conv_y)
         
        conv_z = self.conv3(conv_y)
        conv_z = self.bn3(conv_z)
         
        shortcut_y = self.shortcut1(x)
         
        output_block_1 = self.relu(conv_z + shortcut_y)
         
        # BLOCK 2
        conv_x = self.conv4(output_block_1)
        conv_x = self.bn4(conv_x)
        conv_x = self.relu(conv_x)
         
        conv_y = self.conv5(conv_x)
        conv_y = self.bn5(conv_y)
        conv_y = self.relu(conv_y)
         
        conv_z = self.conv6(conv_y)
        conv_z = self.bn6(conv_z)
         
        shortcut_y = self.shortcut2(output_block_1)
         
        output_block_2 = self.relu(conv_z + shortcut_y)
         
        # BLOCK 3
        conv_x = self.conv7(output_block_2)
        conv_x = self.bn7(conv_x)
        conv_x = self.relu(conv_x)
         
        conv_y = self.conv8(conv_x)
        conv_y = self.bn8(conv_y)
        conv_y = self.relu(conv_y)
         
        conv_z = self.conv9(conv_y)
        conv_z = self.bn9(conv_z)
         
        shortcut_y = self.shortcut3(output_block_2)
         
        output_block_3 = self.relu(conv_z + shortcut_y)
         
        # FINAL
        gap_out = self.global_avg_pool(output_block_3)
         
        gap_out = gap_out.view(gap_out.size(0), -1)
         
        out = self.fc(gap_out)
         
        return out
