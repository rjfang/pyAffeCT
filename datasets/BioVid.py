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

from pyAffeCT.datasets import AffectDataset

class BioVid(AffectDataset):
    
    modality_list = ['ECG','GSR','EMG']
    meta_columns = ['TIMESTAMP','Label']
    sampling_rate = 512
    window_length = 5
    window_shift = 5
    
    # initilization function rewrite
    def __init__(self,root):
        super().__init__(root)
        self.root = root
        self.subject_list = [subject[:-4] for subject in os.listdir(self.root+'/biosignals_raw')]
    
    def read_raw_biodata(self, subject):
        df_raw = pd.read_csv(self.root+'/biosignals_raw/'+subject+'.csv')
        df_split = self._split_biosignals(df_raw)
        return df_split
    
    def _split_biosignals(self, df_raw):
        df = df_raw.copy()
        
        raw_list = df['time	gsr	ecg	emg_trapezius	emg_corrugator	emg_zygomaticus'].values.tolist()
        time_list = []
        gsr_list = []
        ecg_list = []
        emg_trapezius = []


        for i in  range(len(raw_list)):
            time_list.append(raw_list[i].split()[0])
            gsr_list.append(raw_list[i].split()[1])
            ecg_list.append(raw_list[i].split()[2])
            emg_trapezius.append(raw_list[i].split()[3])

        p = pd.date_range(start=datetime.datetime(2000,1,1),\
                          end=datetime.datetime(2000,1,1)+datetime.timedelta(seconds=len(df)//self.sampling_rate),periods=len(df))
        
        df['TIMESTAMP'] = p
        df['GSR'] = pd.to_numeric(gsr_list)
        df['ECG'] = pd.to_numeric(ecg_list)
        df['EMG'] = pd.to_numeric(emg_trapezius)

        del df['time	gsr	ecg	emg_trapezius	emg_corrugator	emg_zygomaticus']
        
        return df
    
    def read_label(self, subject):
        df_label = pd.read_csv(self.root+'/label/'+subject+'.csv')
        df_label = self._split_labels(df_label)
        return df_label
    
    def _split_labels(self, df_raw):
        df = df_raw.copy()
        
        raw_list = df['time\tlabel'].values.tolist()
        time_list = []
        label_list = []


        for i in  range(len(raw_list)):
            time_list.append(raw_list[i].split()[0])
            label_list.append(raw_list[i].split()[1])

        # convert timestamp into ms unit and add a fake time start (2000,1,1)
        df['TIMESTAMP'] = [datetime.datetime(2000,1,1)+datetime.timedelta(milliseconds=int(t)//1000) for t in time_list]
        
        df['Label'] = pd.to_numeric(label_list)

        del df['time\tlabel']
        
        return df
    
    
    def sync_bio_label(self, subject):
        """
        Read synchronized bio data and labels for a subject.

        Parameters
        ----------
        subject : int
            The subject number to read bio+label data.

        Returns
        -------
        df : dataframe
            A dataframe that has synchronized biodata and label like ['TIMESTAMP','ECG','Label'].

        """
        df_bio = self.read_raw_biodata(subject)
        df_label = self.read_label(subject)
        
        # merge the biosignals and labels, based on the timestamp, tolerance 1ms because 1000Hz
        df = pd.merge_asof(df_bio,df_label.sort_values('TIMESTAMP'), on = 'TIMESTAMP', tolerance=pd.Timedelta('100ms'))
        # fill COND NaNs with first values
        df.fillna(method='ffill',inplace = True)
        df.fillna(method='bfill',inplace = True)
        
        df.reset_index(drop=True,inplace=True)
        # assign the individual subject's bio data to class
        self.df_individual = df
        
        return df


    def label_generation(self, df_feature, task, num_classes=2, groupby='proportion', baffle=None):
        df_labeled = df_feature.copy()
        
        # rename label column to 'Label'
        df_labeled.rename(columns={task:'Label'}, inplace=True)
        
        assert num_classes in [2,5]
        
        if num_classes == 2:
            df_labeled = self.remove_rows_on_label(df_labeled, [1,2,3])
            df_labeled['Label'] = df_labeled['Label'].apply(lambda x:0 if x == 0 else 1)
        
        return df_labeled.reset_index(drop=True)
    
