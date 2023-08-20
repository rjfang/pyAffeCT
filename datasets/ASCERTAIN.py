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
from scipy import io
from tqdm import tqdm

from pyAffeCT.datasets import AffectDataset

class ASCERTAIN(AffectDataset):
    
    modality_list = ['ECG_L','ECG_R','GSR']
    self_report_columns = ['Arousal','Valence', 'Engagement', 'Liking', 'Familiarity']
    personality_columns = ['Extroversion','Agreeableness','Conscientiousness','Emotional Stability','Openness']
    meta_columns = ['TIMESTAMP']+self_report_columns + personality_columns
    subject_list = list(range(1,59))
    sampling_rate = 256
    
    # initilization function rewrite
    def __init__(self,root):
        super().__init__(root)
        self.root = root     
        
        if not os.path.exists(os.path.join(self.root,'Processed_data')):
            os.makedirs(os.path.join(self.root,'Processed_data'))
            self.save_sync_data()
    
    def sync_bio_label(self, subject):
        df_subject = pd.read_csv(os.path.join(self.root,'Processed_data',str(subject)+'.csv'))
        return df_subject
    
    def save_sync_data(self):
        print("ASCERTAIN: Processing and Saving data...")
        for subject in tqdm(range(1,59)):
            df_sync = self._sync_bio_label_save(subject)
            df_sync.to_csv(os.path.join(self.root,'Processed_data',str(subject)+'.csv'), index=False)
        
    def _sync_bio_label_save(self, subject):
        
        df_subject = pd.DataFrame()
        
        for clip in range(1,37):
            df_ecg = self._parse_ecg_mat_csv(subject, clip)
            df_gsr = self._parse_gsr_mat_csv(subject, clip)
            self_report = self._parse_self_report(subject, clip)
            personality = self._parse_personality(subject)
            
            # merge the biosignals and labels, based on the timestamp
            df_clip = pd.merge_asof(df_ecg,df_gsr.sort_values('TIMESTAMP'), on = 'TIMESTAMP', tolerance=pd.Timedelta('100ms'))
            df_clip[self.personality_columns] = personality
            df_clip[self.self_report_columns] = self_report
            
            df_subject = pd.concat([df_subject, df_clip], axis = 0)
        
        df_subject.fillna(method='bfill',inplace = True)
        df_subject.fillna(method='ffill',inplace = True)
        
        return df_subject.reset_index(drop=True)
    
    
    
    def _parse_ecg_mat_csv(self,subject_id, clip_id):
        
        subject_id = str(subject_id) if subject_id > 9 else '0'+str(subject_id)
        
        read_path = os.path.join(self.root,'ASCERTAIN_Raw','ECGData','Movie_P'+subject_id,'ECG_Clip' + str(clip_id)+'.mat')
        mat_file = io.loadmat(read_path)
        ecg_data = mat_file['Data_ECG']
        
        # save mat to a csv file
        if int(subject_id) >= 9:
            df_ecg = pd.DataFrame(data = ecg_data[:,[0,4,5]], columns = ['TIMESTAMP', 'ECG_R','ECG_L'])
        else:
            df_ecg = pd.DataFrame(data = ecg_data[:,[0,1,2]], columns = ['TIMESTAMP', 'ECG_R','ECG_L'])
        
        # format timestamp to datetime
        df_ecg['TIMESTAMP'] = df_ecg['TIMESTAMP'].apply(lambda x:datetime.datetime(2000,1,1)+datetime.timedelta(milliseconds=x))
        
        return df_ecg

    
    def _parse_gsr_mat_csv(self, subject_id, clip_id):
        
        subject_id = str(subject_id) if subject_id > 9 else '0'+str(subject_id)
        
        read_path = os.path.join(self.root,'ASCERTAIN_Raw','GSRData','Movie_P'+subject_id,'GSR_Clip' + str(clip_id)+'.mat')
        mat_file = io.loadmat(read_path)
        gsr_data = mat_file['Data_GSR']
        
        df_gsr = pd.DataFrame(data = gsr_data[:,[0,4]], columns = ['TIMESTAMP','GSR'])
        
        # format timestamp to datetime
        df_gsr['TIMESTAMP'] = df_gsr['TIMESTAMP'].apply(lambda x:datetime.datetime(2000,1,1)+datetime.timedelta(milliseconds=x))
        
        return df_gsr

    
    def _parse_self_report(self, subject_id, clip_id):
        read_path = os.path.join(self.root,'ASCERTAIN_Features','Dt_SelfReports.mat')

        mat_file = io.loadmat(read_path)
        
        # [Arousal,Valence, Engagement, Liking, Familiarity]
        ratings = mat_file['Ratings'][:,subject_id-1,clip_id-1]
        
        return ratings
    
    
    def _parse_personality(self, subject_id):
        read_path = os.path.join(self.root,'ASCERTAIN_Features','Dt_Personality.mat')
        
        mat_file = io.loadmat(read_path)['Personality']
        
        # return a 5-d list [a,b,c,d,e]
        return mat_file[subject_id-1]