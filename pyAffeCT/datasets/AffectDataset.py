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
import neurokit2 as nk
from scipy import stats
from tqdm import tqdm
import random
import torch
import torch.utils.data as data
import torch.nn.functional as F

from pyAffeCT.feature_extraction.feature_extraction_lib import *
from pyAffeCT.visualization.plot_lib import *
from pyAffeCT.preprocessing.preprocess_lib import *

import warnings
warnings.filterwarnings('ignore')

class AffectDataset:
    'A parent class of Affective Computing datasets'
    
    subject_list = [1,2]
    meta_columns = ['TIMESTAMP','Label'] # these columns won't be normalized/standardized
    modality_list = []
    modality_included = []
    sampling_rate = 1000
    window_length = 5
    window_shift = window_length
    min_event_window = 1
    max_event_window = 5
    
    
    def __init__(self, root):
        self.root = root
        self.pyeda_enabled = False
        self.pyeda_model_name = self.__class__.__name__+'.t7'
        
    ''' *****************************************************************************
        ************                   Data Reading                     *************
        *****************************************************************************
    ''' 
    def read_raw_biodata(self,subject):
        print('Reserved function')
        return None
    
    def read_label(self, subject):
        print('Reserved function')
        return None
    
    def sync_bio_label(self, subject):
        """
        Generate a synchronized dataframe that contains biosignals and labels.

        Parameters
        ----------
        subject : int
            The subject id.

        Returns
        -------
        df : pd.Dataframe
            A dataframe with columns of e.g., "TIMESTAMP,ECG,...,Label".

        """
        df_bio = self.read_raw_biodata(subject)
        df_label = self.read_label(subject)
        
        # merge the biosignals and labels, based on the timestamp
        df = pd.merge_asof(df_bio,df_label.sort_values('TIMESTAMP'), on = 'TIMESTAMP', tolerance=pd.Timedelta('1s'))
        
        return df
    
    def sync_flatten_data(self, subject, window_length, label_column = 'Label', modality_included=None):
        """
        Read synchronized flatten raw biosignals with label at the end. Typically convienient for end-to-end Deep Learning.

        Parameters
        ----------
        subject : int
            Subject id.
        window_length : int
            Window length in second.
        label_column : str, optional
            The label column. The default is 'Label'.
        modality_included : TYPE, optional
            Signals to include and flatten. The default is None, all modalities will be included.

        Returns
        -------
        df_flatten : pd.DataFrame
            A flattened raw biosignals dataframe with first N columns as second and last column label.

        """
        
        # if signals to include is not specified, use all modalities
        if modality_included is None:
            modality_included = self.modality_list
        
        # raw dataframe
        df_sync = self.sync_bio_label(subject)
        
        # flattened df
        steps = window_length*self.sampling_rate
        columns = list(range(steps*len(modality_included)))
        columns.append('Label')
        
        df_flatten = pd.DataFrame(columns = columns)
        
        index = 0
        while index + steps<len(df_sync):
            if df_sync.loc[index+steps, label_column] == df_sync.loc[index, label_column]:
                data_list = []
                for modality in modality_included:
                    data_list.extend(df_sync.loc[index:index+steps-1, modality])
                data_list.append(df_sync.loc[index+1,label_column])

                df_flatten.loc[len(df_flatten)] = data_list
                
                index += steps
            else:
                # else, step 1 second
                index += self.sampling_rate
        
        # set column names to be string
        df_flatten.columns = df_flatten.columns.astype(str)
        
        return df_flatten
    
    def read_all_flatten_data(self,  window_length, subject_included=None, 
                              label_column = 'Label', modality_included=None):
        """
        

        Parameters
        ----------
        subject_included : list of int
            Subject id to be included.
        window_length : int
            Window length in second.
        label_column : str, optional
            The label column. The default is 'Label'.
        modality_included : TYPE, optional
            Signals to include and flatten. The default is None, all modalities will be included.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # an empty dataframe used to save all subject's data
        df_flatten = pd.DataFrame()
        
        # if no subjects specified, use all subjects in the dataset
        if subject_included is None:
            subject_included = self.subject_list
        
        # loop all subjects and concatenate the data to df_flatten
        for subject in tqdm(subject_included):
            df = self.sync_flatten_data(subject, window_length, label_column, modality_included)
            df_flatten = pd.concat([df_flatten, df], axis=0)
        
        return df_flatten.reset_index(drop=True)
            
    
    def read_all_biodata(self):
        """ Looping all subjects' biodata and concatenate """
        df_bio = pd.DataFrame()
        # loop all subjects' biodata and combie
        for subject in self.subject_list:
            df = self.read_raw_biodata(subject)
            df_bio = pd.concat([df_bio,df],axis=0)
            
        return df_bio.reset_index(drop=True)
    
    # print list of all subjects id
    def print_subject(self):
        print(self.subject_list)
    

    ''' *****************************************************************************
        ************                  Preprocessing                     *************
        *****************************************************************************
    ''' 
    
    def preprocessing_filter(self, df_bio, filter_dict):
        """
        Function to preprocess a dataframe with filters
        inputs:
            filter_dict = {'ECG':{'order':4,'cut_freq':(0.1,200),'btype':'bp','ftype':'butterworth'},
                           'GSR':{'order':10,'cut_freq':10,'btype':'lp','ftype':'butterworth'}}
        """
        df_filtered = df_bio.copy()
        for modality in filter_dict.keys():
            df_filtered[modality] = filter_sub(df_filtered[modality], filter_dict[modality], self.sampling_rate)
        return df_filtered

    
    def visualize_filter(self, df_bio, filter_dict, xrange='7s'):
        if len(df_bio) > 100000:
            df_original = df_bio.copy()[:100000]
            df_filtered = df_bio.copy()[:100000]
        else:
            df_original = df_bio.copy()
            df_filtered = df_bio.copy()
            
        title_list = []
        timestamp=df_original.TIMESTAMP
        for modality in filter_dict.keys():
            df_filtered[modality] = filter_sub(df_filtered[modality], filter_dict[modality], self.sampling_rate)
            title_list.extend([modality, modality+'_filtered'])
        
        filter_plot_sub(timestamp, df_original, df_filtered, filter_dict, title_list, xrange)
        
    
    def remove_rows_on_label(self, df, labels2remove, label_column='Label'):
        df_new = df.copy().reset_index(drop=True)
        for label in labels2remove:
            df_new = df_new.drop(df_new[df_new[label_column]==label].index)
        
        return df_new.reset_index(drop=True)
    
    
    ''' *****************************************************************************
        ************              Feature Extraction                    *************
        *****************************************************************************
    ''' 

    
    def time_window(self, signal, feature_extraction_func, sampling_rate, window_length, window_shift):
        """ Function of generating time window slice and feature extraction"""
        index = 0
        df_feature = pd.DataFrame()
        while index <= len(signal)-window_length*sampling_rate:
            df_temp = feature_extraction_func(signal[index:index+window_length*sampling_rate], sampling_rate)
            index = index+window_shift*sampling_rate
            df_feature = pd.concat([df_feature,df_temp], axis=0)
            
        df_feature.reset_index(drop=True,inplace=True)
        return df_feature
    
    """ Function of generating time window slice based on events"""
    def event_window(self, signal, label, feature_extraction_func, sampling_rate, min_window, max_window):
        index = 0
        df_feature = pd.DataFrame()
        while index <= len(signal) - max_window*sampling_rate:
            label_slice = label[index:index+max_window*sampling_rate]
            # if the label changes during the max window
            if label_slice.nunique() > 1:
                # window_end is the index of the first different label
                window_end = label_slice[label_slice!=label_slice.iloc[0]].index.tolist()[0] 
                if window_end > index + min_window*sampling_rate:
                    signal_slice = signal[index:window_end]
                    df_temp = feature_extraction_func(signal_slice, sampling_rate)
                    df_feature = pd.concat([df_feature,df_temp], axis=0)
                    index = window_end
                else:
                    index = window_end
            else:
                window_end = index+max_window*sampling_rate
                signal_slice = signal[index:window_end]
                df_temp = feature_extraction_func(signal_slice, sampling_rate)
                df_feature = pd.concat([df_feature,df_temp], axis=0)
                index = window_end
        df_feature.reset_index(drop=True,inplace=True)
        return df_feature

    """ Function to extract mode, mostly used for meta-information columns """
# =============================================================================
#     def mode(self, c_slice, sampling_rate):
#         return pd.DataFrame([stats.mode(c_slice)[0][0]],columns=[c_slice.name])
#     
# =============================================================================
    def mode(self, c_slice, sampling_rate):
        if c_slice.isnull().all():
            returnVal = np.nan
        else:
            returnVal = c_slice.value_counts().idxmax()
        return pd.DataFrame([returnVal],columns=[c_slice.name])
    
    def mean(self, signal, sampling_rate):
        returnVal = np.mean(signal)
        
        return pd.DataFrame([returnVal], columns=[signal.name])
        
    """ Feature Extraction for ECG, by NK2 """
    def ECG_feature(self, ecg, sampling_rate):
        try:
            ecg_cleaned = nk.ecg_clean(ecg,sampling_rate = sampling_rate)
            #nk.ecg_plot(signals, sampling_rate=700)
            peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            df_hrv = nk.hrv_time(peaks, sampling_rate=sampling_rate)
        except:
            df_hrv = pd.DataFrame()
        return df_hrv
    
    """ Feature Extraction for GSR """
    def GSR_feature(self, gsr, sampling_rate):
        df_gsr = pd.DataFrame(columns=generate_GSR_feature_list())
        df_gsr.loc[0] = generate_GSR_features(gsr)
        if self.pyeda_enabled:
            df_pyeda = pd.DataFrame(columns=generate_pyEDA_feature_list())
            df_pyeda.loc[0] = pyEDA_process(gsr,self.pyeda_model_name)
            df_gsr = pd.concat([df_gsr, df_pyeda], axis=1)
        return df_gsr
    
    """ Feature Extraction for EMG """
    def EMG_feature(self, emg, sampling_rate):
        df_emg = pd.DataFrame(columns=generate_EMG_feature_list())
        df_emg.loc[0] = generate_EMG_features(emg)
        return df_emg
    
    def enable_pyeda(self):
        self.pyeda_enabled=True

    """ A process function to extract all features for one subject """
    def process_feature(self, df_sync, method = 'fixed', label_column = 'Label'):
        # read synced data
        df = df_sync.copy()
        df_feature = pd.DataFrame()
        
        for column in df.columns:
            if column == 'ECG' or column == 'EKG':
                func = self.ECG_feature
            elif column =='GSR':
                func = self.GSR_feature
            elif column == 'EMG':
                func = self.EMG_feature
            elif column in self.meta_columns:
                func = self.mode
            else:
                func = self.mean
            
            if method == 'fixed':
                df_temp = self.time_window(df[column], func, self.sampling_rate, self.window_length, self.window_shift)
            elif method == 'event':
                label = df[label_column]
                df_temp = self.event_window(df[column], label, func, self.sampling_rate, self.min_event_window, self.max_event_window)
            
            df_feature = pd.concat([df_feature,df_temp], axis=1)
            
        return df_feature
    

    def read_all_biofeature(self):
        df_all = pd.DataFrame()
        # loop all subjects' biodata and combie
        for subject in self.subject_list:
            df_bio=self.sync_bio_label(subject)
            df = self.process_feature(df_bio)
            df_all = pd.concat([df_all,df],axis=0)
            
        return df_all
    
    def prepare_GSR(self):
        df_all = pd.DataFrame()
        for subject in self.subject_list[:10]:
            df = self.sync_bio_label(subject)
            df_all = pd.concat([df_all,df],axis=0)
        df_gsr = df_all['GSR'].values
        
        #df_gsr.reshape(self.sampling_rate*self.window_length)
        if self.pyeda_enabled:
            pyEDA_train(df_gsr[:self.sampling_rate*5], self.sampling_rate, self.window_length, self.pyeda_model_name)
    
    ''' *****************************************************************************
        ************                 Postprocessing                     *************
        *****************************************************************************
    ''' 
    
    def drop_nan(self, df, threshold = 0.5):
        """
        To drop NaN values from a dataframe by y-axis
        inputs:
            threshold: a fraction of the proportion of minimum nan values to drop the entire column
        """
        df_wo_nan = remove_nan(df,threshold)
        return df_wo_nan
    
    def normalize(self, df):
        """ Normalization for an entire df, but will ignore meta_columns """
        df = df.copy()
        for i in list(df.columns):
            if i not in self.meta_columns:
                Max = np.max(df[i])
                Min = np.min(df[i])
                if Max == Min:
                    df[i] = 0.5
                else:
                    df[i] = (df[i] - Min)/(Max - Min)
        return df
    
    
    def standardize(self, df):
        """ Standardization for an entire df, but will ignore meta_columns """
        df = df.copy()
        for i in list(df.columns):
            if i not in self.meta_columns:
                mean = np.mean(df[i])
                std = np.std(df[i])
                
                df[i] = (df[i] - mean)/std
        return df
    
    
    def label_generation(self, df_feature, task, num_classes=2, groupby='proportion', baffle=None):
        """
        Function: to generate label column for a df
        num_classes: number of classes
        task: the name of the target label column
        groupby: how to transform the task column to label, 'proportion' or 'absolute', 
                    if 'proportion', baffle needs to be provided
        baffle: the length of the baffle needs to be num_classes-1, used to cut the task column
        """
        
        df_labeled = df_feature.copy()
        
        # remove unused label columns
        for col2drop in self.meta_columns:
            if col2drop != task and col2drop in df_labeled.columns:
                del df_labeled[col2drop]
        
        # rename label column to 'Label'
        df_labeled.rename(columns={task:'Label'}, inplace=True)
        
        # drop nan for label column
        df_labeled.dropna(subset=['Label'], inplace=True)
        label_column = df_labeled['Label']
        
        # calculates unique values of the task column without nan
        unique_values = df_labeled.Label.unique()[~np.isnan(df_labeled.Label.unique())]
        if len(unique_values) < num_classes:
            return pd.DataFrame(columns = df_labeled.columns)

        
        if groupby == 'proportion':
            baffle = [np.percentile(sorted(label_column), (baf+1)*100/num_classes) for baf in range(num_classes-1)]

        elif groupby == 'absolute':
            assert len(baffle) == num_classes-1
            baffle = baffle
        
        # insert infinities to first and last
        baffle.insert(0,-np.inf)
        baffle.append(np.inf)
        
        
        for i in range(len(baffle)-1):
            label_column[label_column.between(baffle[i],baffle[i+1],inclusive='right')] = i

        df_labeled['Label'] = label_column
        
        return df_labeled
    
    
    ''' *****************************************************************************
        ************                 Visualization                      *************
        *****************************************************************************
    ''' 
    def ecg_plot(self, subject, label=None):
        df_sync = self.sync_bio_label(subject)
        if label!=None:
            label = df_sync[label] 
        ecg_plot_sub(df_sync['TIMESTAMP'], df_sync['ECG'], label, sampling_rate = 1000)
        
    def gsr_plot(self, subject, label=None):
        df_sync = self.sync_bio_label(subject)
        if label!=None:
            label = df_sync[label] 
        gsr_plot_sub(df_sync['TIMESTAMP'], df_sync['GSR'], label, sampling_rate = 1000)
    

    def feature_plot(self, df, feature_name):
        """
        The function to plot features in time-seires
        inputs:
            df: dataframe
            feature_name: str or list of str, the feature/s to plot
        """
        time_s = pd.date_range('2000/11/11', periods=len(df), freq=str(self.window_shift)+'s')
        df['TIMESTAMP'] = pd.to_datetime(time_s, format='%H:%M:%S')
        if 'Label' in df.columns:
            feature_plot_sub(df['TIMESTAMP'], df[feature_name], df['Label'])
        else:
            feature_plot_sub(df['TIMESTAMP'], df[feature_name])
    
    def feature_heatmap(self, df_feature):
        """
        The heatmap of features and the label
        """
        heat_map(df_feature)
    
    
    
    ''' *****************************************************************************
        ************              Data End-to-end Export                *************
        *****************************************************************************
    ''' 
    
    def prepare_dataframe(self, modality_included=None, subject_included=None, 
                          isSplit=True, split_by_subject = True, test_rate=0.3,
                          filter_dict=None,feature_method='fixed',feature_label_column='Label',
                          drop_nan_thres=0.5,isNormalize=True, isStandardize=False, window_length=5,
                          window_shift=5,label_task='Label', num_classes=2, groupby='proportion', baffle=None, 
                          isResample=True):
        """
        Function to prepare a dataframe which is ready for classification tasks
        inputs:
            modality_included: a list of modalities intended to use
            subject_included: a list of subjects intended to include
            isSplit: whether to split the dataframe to train and test
            split_by_subject: to split by subjects or by samples
            test_rate: test rate, by default 0.3
            filter_dict: the filter dictionary, refer to preprocessing_filter
            feature_method: 'fixed' or 'event' based feature extraction strategy
            feature_label_column: the label column or 'event', if 'event' based feature method is chosen
            drop_nan_thres: drop NaN values threshold, a fraction, 0 if don't drop.
            isNormalize: boolean, normalize the dataframe at subject level
            isStandardize: boolean, standardize the dataframe at subject level
            label_task: which column to be used as label
            num_classes: number of classess to classify
            groupy: the approach to group labels, by 'proportion' or by 'absolute' values
            baffle: if 'absolute' method chosen, baffle needs to be provided, a list of values
            isResample: whether to resample the data to be balanced
        """
        # set default subjects and modalities
        if modality_included is None:
            modality_included = self.modality_list
        if subject_included is None:
            subject_included = self.subject_list
        
        self.window_length = window_length
        self.window_shift = window_shift
            
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        df_all = pd.DataFrame()
        
        if isSplit:
            #random.shuffle(subject_included)
            test_rate = test_rate
        else:
            test_rate = 0
        
        self.prepare_GSR()
        
        for subject in tqdm(subject_included):
            # read synchronized dataset
            df_sync = self.sync_bio_label(subject)
            
            # remove unused label columns
            for column in self.modality_list:
                if column not in modality_included:
                    del df_sync[column]
                    
            # filter data
            if filter_dict is not None:
                df_sync = self.preprocessing_filter(df_sync,filter_dict)
            
            # feature extraction
            df_feature = self.process_feature(df_sync, method=feature_method, label_column=label_task)
            
            # normalize and standardize
            if isNormalize:
                df_feature_rescaled = self.normalize(df_feature)
            elif isStandardize:
                df_feature_rescaled = self.standardize(df_feature)
            else:
                df_feature_rescaled = df_feature.copy()
            

            # label generation
            df_labeled = self.label_generation(df_feature_rescaled, task=label_task,num_classes=num_classes,groupby=groupby,baffle=baffle)
            
            # remove timestamp column if it exists
            if 'TIMESTAMP' in df_labeled.columns:
                del df_labeled['TIMESTAMP']
            
            if split_by_subject:
                if subject_included.index(subject)/len(subject_included) > test_rate:
                    df_train = pd.concat([df_train,df_labeled], axis=0)
                else:
                    df_test = pd.concat([df_test, df_labeled], axis=0)
            else:
                df_all = pd.concat([df_all, df_labeled], axis=0)
        
        if not split_by_subject:
            df_all.reset_index(drop=True,inplace=True)
            df_train = df_all.sample(frac=1-test_rate, axis=0)
            df_test = df_all[~df_all.index.isin(df_train.index)]
            
        
        # drop NaN values
        df_train = self.drop_nan(df_train, threshold=drop_nan_thres)
        if isResample:
            df_train = resample(df_train)
        df_train.reset_index(drop=True,inplace=True)
        
        df_test = self.drop_nan(df_test, threshold=drop_nan_thres)
        if isResample and isSplit:
            df_test = resample(df_test)
        df_test.reset_index(drop=True,inplace=True)
        
        return df_train, df_test
    
    
    def df_to_dataloader(self, df_train, df_test, num_classes, batch_size, isFlatten=False):
        
        # just to use self
        self.dummy = 0
        
        if isFlatten:
            num_channels = 1
        else:
            num_channels = len(self.modality_included)
        
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_test = df_test.sample(frac=1).reset_index(drop=True)
        
        X_train = df_train.loc[:,df_train.columns!='Label']
        y_train = df_train['Label']
        X_test = df_test.loc[:,df_test.columns!='Label']
        y_test = df_test['Label']
        
        
        
        X_train = torch.tensor(X_train.values,dtype=torch.float32).reshape((X_train.shape[0],num_channels,X_train.shape[-1]//num_channels))
        y_train = torch.tensor(y_train.values,dtype=torch.float32)
        y_train = F.one_hot(y_train.long(), num_classes=num_classes)
        
        X_test = torch.tensor(X_test.values,dtype=torch.float32).reshape((X_test.shape[0],num_channels,X_test.shape[-1]//num_channels))
        y_test = torch.tensor(y_test.values,dtype=torch.float32)
        y_test = F.one_hot(y_test.long(), num_classes=num_classes)
        
        train_loader = data.DataLoader(data.TensorDataset(*(X_train,y_train)), batch_size = batch_size)
        test_loader = data.DataLoader(data.TensorDataset(*(X_test,y_test)), batch_size = batch_size)
        
        return train_loader, test_loader, X_train.shape
        
    
    def prepare_flatten_dataframe(self, modality_included=None, subject_included=None, window_length=5,
                          window_shift=5, isSplit=True, split_by_subject = True, test_rate=0.3,
                          filter_dict=None,isNormalize=True, isStandardize=False, 
                          label_task='Label', num_classes=2, groupby='proportion', baffle=None, 
                          isResample=True):
        """
        

        Parameters
        ----------
        modality_included : TYPE, optional
            DESCRIPTION. The default is None.
        subject_included : TYPE, optional
            DESCRIPTION. The default is None.
        window_length : TYPE, optional
            DESCRIPTION. The default is 10.
        isSplit : TYPE, optional
            DESCRIPTION. The default is True.
        split_by_subject : TYPE, optional
            DESCRIPTION. The default is True.
        test_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.
        filter_dict : TYPE, optional
            DESCRIPTION. The default is None.
        isNormalize : TYPE, optional
            DESCRIPTION. The default is True.
        isStandardize : TYPE, optional
            DESCRIPTION. The default is False.
        label_task : TYPE, optional
            DESCRIPTION. The default is 'Label'.
        num_classes : TYPE, optional
            DESCRIPTION. The default is 2.
        groupby : TYPE, optional
            DESCRIPTION. The default is 'proportion'.
        baffle : TYPE, optional
            DESCRIPTION. The default is None.
        isResample : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        
        # set default subjects and modalities
        if modality_included is None:
            self.modality_included = self.modality_list
        else:
            self.modality_included = modality_included
        if subject_included is None:
            subject_included = self.subject_list
        
        self.window_length = window_length
        self.window_shift = window_shift
            
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        df_all = pd.DataFrame()
        
        if isSplit:
            test_rate = test_rate
        else:
            test_rate = 0
        
        for subject in tqdm(subject_included):
            # read synchronized dataset
            df_sync = self.sync_flatten_data(subject, self.window_length, label_task, self.modality_included)
            
            
            # filter data
            if filter_dict is not None:
                df_sync = self.preprocessing_filter(df_sync,filter_dict)
            
# =============================================================================
#             # normalize and standardize
#             if isNormalize:
#                 df_feature_rescaled = self.normalize(df_sync)
#             elif isStandardize:
#                 df_feature_rescaled = self.standardize(df_sync)
#             else:
#                 df_feature_rescaled = df_feature.copy()
# =============================================================================
            
            # label generation
            df_labeled = self.label_generation(df_sync, task=label_task,num_classes=num_classes, groupby=groupby, baffle=baffle)
            
            # remove timestamp column if it exists
            if 'TIMESTAMP' in df_labeled.columns:
                del df_labeled['TIMESTAMP']
                
            if split_by_subject:
                if subject_included.index(subject)/len(subject_included) > test_rate:
                    df_train = pd.concat([df_train,df_labeled], axis=0)
                else:
                    df_test = pd.concat([df_test, df_labeled], axis=0)
            else:
                df_all = pd.concat([df_all, df_labeled], axis=0)
        
        if not split_by_subject:
            df_all.reset_index(drop=True,inplace=True)
            df_train = df_all.sample(frac=1-test_rate, axis=0)
            df_test = df_all[~df_all.index.isin(df_train.index)]
            
        
        if isResample:
            df_train = resample(df_train)
        df_train.reset_index(drop=True,inplace=True)
        
        if isResample and isSplit:
            df_test = resample(df_test)
        df_test.reset_index(drop=True,inplace=True)
        
        return df_train, df_test