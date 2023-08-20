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
from tqdm import tqdm

from pyAffeCT.datasets import AffectDataset

class BioVidEmo(AffectDataset):
    
    modality_list = ['ECG','GSR','EMG']
    meta_columns = ['TIMESATMP','Label']
    emotion_dic = {'amusement':0, 'anger':1, 'disgust':2, 'fear':3, 'sad':4}
    sampling_rate = 512
    window_length = 5
    window_shift = window_length
    
    # initilization function rewrite
    def __init__(self,root):
        super().__init__(root)
        self.root = root
        self.split_path = os.path.join(os.path.dirname(self.root) , 'bio_split')
        
        if not os.path.exists(self.split_path):
            self._split_files_by_subject()
        
        self.subject_list = os.listdir(self.split_path)
    
    def _split_files_by_subject(self):
        file_list = os.listdir(self.root)
        
        # for every unsplit files
        print('BioVid Emo: Preparing and splitting raw files...')
        for file in tqdm(file_list):
            #print(file)
            
            subject_id,_,age,_ = self._parse_emotion_file_name(file)
            
            folder = os.path.exists(self.split_path+'/'+str(subject_id))
            
            while folder and (age != int(os.listdir(os.path.join(self.split_path, str(subject_id)))[0][12:14]) \
                              or file[:9] != os.listdir(os.path.join(self.split_path, str(subject_id)))[0][:9]):
                subject_id += 1 # use another subject_id
                folder = os.path.exists(self.split_path+'/'+str(subject_id)) # re-find folder
                
            if not folder:
                os.makedirs(self.split_path+'/'+str(subject_id))
            
            
            # read dataframe
            df_unsplit = pd.read_csv(self.root+'/'+file)
            
            # split dataframe
            #df_split = split_biosignals(df_unsplit)
            
            df_unsplit.to_csv(self.split_path+"/"+str(subject_id)+"/"+file, index = False)
        
    def _parse_emotion_file_name(self,file_name):
        subject_id = int(file_name[0:6])
        gender = 0 if file_name[10:11] == 'w' else 1
        age = int(file_name[12:14])
        emotion = file_name[15:-8]
        
        return subject_id, gender, age,emotion
    
    def sync_bio_label(self, subject):
        
        df_subject=pd.DataFrame()
        
        path = os.path.join(self.split_path, subject)
        for csv_file in os.listdir(path):
            subject_id, gender,age,emotion = self._parse_emotion_file_name(csv_file)
            df = pd.read_csv(os.path.join(path,csv_file))
            df.columns = ['GSR','ECG','EMG']
            df['Label'] = emotion
            df.replace(self.emotion_dic,inplace=True)
            df_subject = pd.concat([df_subject, df], axis=0)
        
        p = pd.date_range(start=datetime.datetime(2000,1,1),\
                          end=datetime.datetime(2000,1,1)+datetime.timedelta(seconds=len(df_subject)//self.sampling_rate),periods=len(df_subject))
        df_subject.insert(0,'TIMESTAMP',p)
        
        return df_subject.reset_index(drop=True)
        
