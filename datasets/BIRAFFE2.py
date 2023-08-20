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


class BIRAFFE2(AffectDataset):
    '''BIRAFFE2 dataset: Kutt, Krzysztof, et al. "BIRAFFE2, a multimodal dataset for emotion-based personalization in rich affective game environments." Scientific Data 9.1 (2022): 1-15.'''
    modality_list = ['ECG','GSR']
    meta_columns = ['TIMESTAMP', 'ID', 'COND', 'IADS-ID', 'IAPS-ID', 'Valence', 'Arousal', 'ANS-TIME']
    
    # initilization function rewrite
    def __init__(self,root):
        super().__init__(root)
        self.root = root
        df_meta = pd.read_csv(root+'/BIRAFFE2-metadata.csv', sep=';')
        #df_meta = df_meta.dropna()
        self.df_meta = df_meta
        self.subject_list = df_meta['ID'].values.tolist()
        self.subject_list.remove(723) # missing files for 723
    
    # read raw biosignal for a subject
    def read_raw_biodata(self, subject):
        """
        Read raw biosignal data from the dataset

        Parameters
        ----------
        subject : the subject number to extract data, must come from self.subject_list

        Returns
        -------
        df_bio : a dataframe which contains 'TIMESTAMP', 'ECG' and 'GSR' and other necessary modalities

        """
        
        df_bio = pd.read_csv(self.root+'/BIRAFFE2-biosigs/SUB'+str(subject)+'-BioSigs.csv')
        df_bio['TIMESTAMP'] = pd.to_datetime(df_bio['TIMESTAMP'], unit='s')
        df_bio.rename(columns={'EDA':'GSR'},inplace=True)
        return df_bio
    
    def read_label(self, subject):
        """
        Read label of the dataset 

        Parameters
        ----------
        subject : int
            The subject number to read label.

        Returns
        -------
        df_label : dataframe
            the dataframe that contains ['TIMESATMP','Labels', etc].

        """
        df_label = pd.read_csv(self.root+'/BIRAFFE2-procedure/SUB'+str(subject)+'-Procedure.csv', delimiter = ';')
        df_label['TIMESTAMP'] = pd.to_datetime(df_label['TIMESTAMP'], unit='s')
        df_label['COND'].fillna(df_label['EVENT'],inplace=True) # merge COND and EVENT, they are complementary
        del df_label['EVENT']
        df_label.rename(columns={'ANS-VALENCE':'Valence','ANS-AROUSAL':'Arousal'}, inplace=True)
        self.meta_columns = df_label.columns.values.tolist() # update meta columns which won't be used in normalization
        
        return df_label
    
    def sync_bio_label(self, subject):
        """
        Read synchronized bio data and label(labels) for a subject.

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
        df_label.fillna('NaN', inplace = True)
        
        # merge the biosignals and labels, based on the timestamp, tolerance 1ms because 1000Hz
        df = pd.merge_asof(df_bio,df_label.sort_values('TIMESTAMP'), on = 'TIMESTAMP', tolerance=pd.Timedelta('1ms'))
        # fill COND NaNs with first values
        df.fillna(method='ffill',inplace = True)
        
        df.replace('NaN',np.nan, inplace = True)
        
        df.reset_index(drop=True,inplace=True)
        # assign the individual subject's bio data to class
        self.df_individual = df
        
        return df
    

    

if __name__ == '__main__':
    biraffe2 = BIRAFFE2('../Data/01 BIRAFFE2')
    df = biraffe2.sync_bio_label(170)
    df_feature = biraffe2.process_feature(170)
    
    #df, info = nk.ecg_process(df["ECG"][1250000:1255000], sampling_rate=1000)
    #nk.ecg_intervalrelated(df, sampling_rate=1000)
    #ecg_cleaned = nk.ecg_clean(df["ECG"], sampling_rate=1000)

    #quality = nk.ecg_quality(ecg_cleaned, sampling_rate=1000)
    #df.to_csv('../01 Data/01 BIRAFFE2/df.csv')
    
    
    
    
    
    