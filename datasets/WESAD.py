# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 rjfang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import pandas as pd
import numpy as np
import datetime
import pickle

from pyAffeCT.datasets import AffectDataset

class WESAD(AffectDataset):
    
    modality_list = ['ECG','GSR','EMG','ACC_1','ACC_2','ACC_3','ST','Resp']
    meta_columns = ['TIMESTAMP','Label']
    
    # initilization function rewrite
    def __init__(self,root):
        super().__init__(root)
        self.root = root
        self.subject_list = list(range(2,18))
        self.subject_list.remove(12)
        self.sampling_rate = 700
        
    # read raw biosignal for a subject
    def sync_bio_label(self, subject):
        """
        Read raw biosignal data from the dataset

        Parameters
        ----------
        subject : the subject number to extract data, must come from self.subject_list

        Returns
        -------
        df_bio : a dataframe which contains 'TIMESTAMP', 'ECG' and 'GSR'

        """
        
        pkl_file = self._load_data(subject)
        df_bio = self._pkl2csv(pkl_file)
        
        # add a fake timestamp column to the left, fake time starts from 2000-1-1
        p = pd.date_range(start=datetime.datetime(2000,1,1),\
                          end=datetime.datetime(2000,1,1)+datetime.timedelta(seconds=len(df_bio)//self.sampling_rate),periods=len(df_bio))
        df_bio.insert(0,'TIMESTAMP',p)
        
        return df_bio    
    
    def _load_data(self, subject):
        """Given path and subject, load the data of the subject"""
        subject = '/S'+str(subject)
        file_path = self.root+subject+subject+'.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        return data


    def _pkl2csv(self, pkl_file):
        
        df_C = pd.DataFrame()
        #df['ACC_C'] = pkl_file['signal']['chest']['ACC']
        df_C['ACC_1'] = pkl_file['signal']['chest']['ACC'][:,0]
        df_C['ACC_2'] = pkl_file['signal']['chest']['ACC'][:,1]
        df_C['ACC_3'] = pkl_file['signal']['chest']['ACC'][:,2]
        

        df_C['ECG'] = pkl_file['signal']['chest']['ECG'].flatten()
        df_C['EMG'] = pkl_file['signal']['chest']['EMG'].flatten()
        df_C['GSR'] = pkl_file['signal']['chest']['EDA'].flatten()
        df_C['ST'] = pkl_file['signal']['chest']['Temp'].flatten()
        df_C['Resp'] = pkl_file['signal']['chest']['Resp'].flatten()
        
        df_C['Label'] = pkl_file['label'].flatten()
        
        return df_C
    
    def label_generation(self, df_feature, task, num_classes=2, groupby='proportion', baffle=None):
        df_labeled = df_feature.copy()
        
        # rename label column to 'Label'
        df_labeled.rename(columns={task:'Label'}, inplace=True)
        
        df_labeled = self.remove_rows_on_label(df_labeled, [0,3,4,5,6,7])
        
        df_labeled['Label'] = df_labeled['Label'].apply(lambda x:0 if x==1 else 1)
        
        return df_labeled