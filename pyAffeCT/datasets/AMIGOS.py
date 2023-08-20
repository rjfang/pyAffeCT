# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
import datetime
import pickle

from pyAffeCT.datasets import AffectDataset

class AMIGOS(AffectDataset):
    
    modality_list = ['ECG','GSR']
    meta_columns = ['TIMESATMP','Label']
    VIDEO_NUM = 16
    
    # initilization function rewrite
    def __init__(self,root):
        super().__init__(root)
        self.root = root
        self.subject_list = list(range(1,41))
        self.subject_list.remove(9, 12, 21, 22, 23, 24, 33)
        self.sampling_rate = 128
        
    # read raw biosignal for a subject
    def sync_bio_label(self, subject):
        ''' Read AMIGOS dataset '''
        amigos_data = np.array([])
        sid = subject
        for vid in range(self.VIDEO_NUM):
            signals = np.genfromtxt(os.path.join(self.root, "{}_{}.csv".format(sid + 1, vid + 1)),
                                    delimiter=',')
            
            ecg_signals = signals[:, 14]  # Column 14 or 15
            gsr_signals = signals[20:, -1]  # ignore the first 20 data, since there is noise in it

            amigos_data = np.vstack((amigos_data, ecg_signals,gsr_signals)) if amigos_data.size \
                else np.vstack((ecg_signals,gsr_signals))

        return amigos_data
    
    def read_labels(self, path, subject):
        """ Read labels of arousal and valance
        Arguments:
            path: path of the label file
        Return:
            a_labels: arousal labels
            v_labels: valance labels
        """
        labels = np.loadtxt(path, delimiter=',')
        labels = labels[:, :2]
        a_labels, v_labels = [], []

        a_labels_mean = np.mean(labels[subject * 16:subject * 16 + 16, 0])
        v_labels_mean = np.mean(labels[subject * 16:subject * 16 + 16, 1])
        for idx, label in enumerate(labels[subject * 16:subject * 16 + 16, :]):
            a_tmp = 1 if label[0] > a_labels_mean else 0
            v_tmp = 1 if label[1] > v_labels_mean else 0
            a_labels.append(a_tmp)
            v_labels.append(v_tmp)
        a_labels, v_labels = np.array(a_labels), np.array(v_labels)
    
        return a_labels, v_labels