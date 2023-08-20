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
from xml.dom.minidom import parse
import pyedflib
from tqdm import tqdm

from pyAffeCT.datasets import AffectDataset

class HCI_Tagging(AffectDataset):
    
    modality_list = ['ECG1','ECG2','ECG3','GSR','RESP','TEMP']
    meta_columns = ['TIMESTAMP','Arousal','Valence','Control','Predictability','Emotion','Tag']
    sampling_rate = 256

    tag_label = {'disgust': 0, 'sadness': 1, 'amusement': 2, 'joy': 3, 'fear': 4, 'neutral':5}
    
    tag_label_char = {'69.avi': tag_label['disgust'],
                  '55.avi': tag_label['sadness'],
                  '58.avi': tag_label['amusement'],
                  'earworm_f.avi': tag_label['disgust'],
                  '53.avi':	tag_label['amusement'],
                  '80.avi':	tag_label['joy'],
                  '52.avi':	tag_label['amusement'],
                  '79.avi':	tag_label['joy'],
                  '73.avi':	tag_label['fear'],
                  '90.avi':	tag_label['joy'],
                  '107.avi': tag_label['fear'],
                  '146.avi': tag_label['sadness'],
                  '30.avi':	tag_label['fear'],
                  '138.avi': tag_label['sadness'],
                  'newyork_f.avi': tag_label['neutral'],
                  '111.avi': tag_label['sadness'],
                  'detroit_f.avi': tag_label['neutral'],
                  'cats_f.avi': tag_label['joy'],
                  'dallas_f.avi': tag_label['neutral'],
                  'funny_f.avi': tag_label['joy']
                  }
    
    media_label = {'69.avi': 'disgust',
                  '55.avi': 'sadness',
                  '58.avi': 'amusement',
                  'earworm_f.avi': 'disgust',
                  '53.avi':	'amusement',
                  '80.avi':	'joy',
                  '52.avi':	'amusement',
                  '79.avi':	'joy',
                  '73.avi':	'fear',
                  '90.avi':	'joy',
                  '107.avi': 'fear',
                  '146.avi': 'sadness',
                  '30.avi':	'fear',
                  '138.avi': 'sadness',
                  'newyork_f.avi': 'neutral',
                  '111.avi': 'sadness',
                  'detroit_f.avi': 'neutral',
                  'cats_f.avi': 'joy',
                  'dallas_f.avi': 'neutral',
                  'funny_f.avi': 'joy'
                  }
    
    emotion_felt_label = {0 : 'neutral',
                          1 : 'anger',
                          2 : 'disgust',
                          3 : 'fear',
                          4 : 'joy',
                          5 : 'sadness',
                          6 : 'surprise',
                          7 : 'scream',
                          8 : 'bored',
                          9 : 'sleepy',
                          10: 'unknown',
                          11: 'amusement',
                          12: 'anxiety'
                          
        }
    
    # initilization function rewrite
    def __init__(self,root):
        super().__init__(root)
        self.root = root
        
        self.data_path = os.path.join(self.root, 'Processed_data')
        if not os.path.exists(self.data_path):
            self._parse_save()
        
        self.subject_list = [int(subject) for subject in os.listdir(self.data_path)]
    
    def sync_bio_label(self, subject):
        df_subject = pd.DataFrame()
        for file in os.listdir(os.path.join(self.data_path,str(subject))):
            df_clip = pd.read_csv(os.path.join(self.data_path,str(subject),file))
            df_subject = pd.concat([df_subject, df_clip], axis=0)
            
        return df_subject.reset_index(drop=True)
        
    def _parse_save(self):
        DataPath = os.path.join(self.root, 'Sessions')
        SavePath = self.data_path
        
        print("HCI Tagging: Processing and saving data...")
        for session in tqdm(os.listdir(DataPath)):

            file_path = os.path.join(DataPath , session , 'session.xml')
        
        
            xml = parse(file_path)
            r = xml.documentElement
            subject_id = r.getElementsByTagName('subject')[0].getAttribute('id')
            arousal = r.getAttribute('feltArsl')
            valence = r.getAttribute('feltVlnc')
            control = r.getAttribute('feltCtrl')
            Pred = r.getAttribute('feltPred')
            emotion = r.getAttribute('feltEmo')
            
            media = r.getAttribute('mediaFile')
            sub = r.getElementsByTagName('subject')[0].getAttribute('id')
            
            
            for file in os.listdir(os.path.join(DataPath, session)):
              if file.endswith('.bdf'):
                s=[_ for _ in os.listdir(os.path.join(DataPath, session)) if _.endswith('.bdf')]
                bdf_path = os.path.join(DataPath , session , s[0])
                with pyedflib.EdfReader(bdf_path) as f:
                  n = f.signals_in_file
                  signal_labels = f.getSignalLabels()
                  s = f.getNSamples()[0]
                  
                  # split the valid range of bio data
                  stat = f.readSignal(46)
                  valid_range = [np.argmax(stat),np.argmax(stat)+np.argmax(stat[np.argmax(stat)+100:-1])]
                  
                  # read physiological data
                  ecg1 = f.readSignal(32)[valid_range[0]:valid_range[1]]
                  ecg2 = f.readSignal(33)[valid_range[0]:valid_range[1]]
                  ecg3 = f.readSignal(34)[valid_range[0]:valid_range[1]]
                  gsr1 = f.readSignal(40)[valid_range[0]:valid_range[1]]
                  resp = f.readSignal(44)[valid_range[0]:valid_range[1]]
                  temp = f.readSignal(45)[valid_range[0]:valid_range[1]]
                  
                  # generate data frame
                  bio = [ecg1,ecg2,ecg3,gsr1, resp, temp]
                  bio = np.array(bio).T
                  
        
                  df_bio = pd.DataFrame(bio, columns = ['ECG1','ECG2','ECG3','GSR','RESP','TEMP'])
                  df_bio['Arousal'] = (arousal)
                  df_bio['Valence'] = (valence)
                  df_bio['Control'] = (control)
                  df_bio['Predictability'] = (Pred)
                  df_bio['Emotion'] = int(emotion)
                  df_bio['Tag'] = self.tag_label[self.media_label[media]]
                  # add a fake timestamp column to the left, fake time starts from 2000-1-1
                  p = pd.date_range(start=datetime.datetime(2000,1,1),\
                                    end=datetime.datetime(2000,1,1)+datetime.timedelta(seconds=len(df_bio)//self.sampling_rate),periods=len(df_bio))
                  df_bio.insert(0,'TIMESTAMP',p)
                  
                  folder = os.path.exists(os.path.join(SavePath, str(subject_id)))
                  if not folder:
                      os.makedirs(os.path.join(SavePath, str(subject_id)))
                  
                  df_bio.to_csv(os.path.join(SavePath, str(subject_id), session+'_'+media+'.csv'),index = False)
    
    
