# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:46:03 2023

@author: rjfan
"""
from pyAffeCT.datasets import AffectDataset

class Non_EEG(AffectDataset):
    
    def __init__(self, root):
        super().__init__(root)
        