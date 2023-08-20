# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 rjfang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import pandas as pd
from scipy import signal
from imblearn.under_sampling import RandomUnderSampler

def remove_nan(df,threshold = 0.5):
    df = df.copy()
    df = df.dropna(axis=1, thresh=int(threshold*len(df)))
    
    df = df.dropna(axis=0, how ='any')

    return df

"""
filter_dict = {'ECG':{'order':4,'cut_freq':(0.1,200),'btype':'bp','ftype':'butterworth'},
               'GSR':{'order':10,'cut_freq':10,'btype':'lp','ftype':'butterworth'}}
"""
def filter_sub(biosignal, filter_dict, sampling_frequency):
    if filter_dict['ftype'] == 'butterworth':
        sos = signal.butter(filter_dict['order'], filter_dict['cut_freq'], filter_dict['btype'], fs = sampling_frequency, output='sos')
        filtered = signal.sosfilt(sos, biosignal)
    elif filter_dict['ftype'] == 'chebyshev':
        sos = signal.cheby1(filter_dict['order'], filter_dict['rp'], filter_dict['cut_freq'], filter_dict['btype'], fs=sampling_frequency, output='sos')
        filtered = signal.sosfilt(sos, biosignal)
    elif filter_dict['ftype'] == 'elliptic':
        sos = signal.ellip(filter_dict['order'], filter_dict['rp'], filter_dict['rs'], filter_dict['cut_freq'], filter_dict['btype'], fs=sampling_frequency, output='sos')
        filtered = signal.sosfilt(sos, biosignal)
    elif filter_dict['ftype'] == 'bessel':
        sos = signal.bessel(filter_dict['order'], filter_dict['cut_freq'], filter_dict['btype'], fs=sampling_frequency, output='sos')
        filtered = signal.sosfilt(sos, biosignal)
    elif filter_dict['ftype'] == 'fir':
        taps = signal.firwin(filter_dict['num_taps'], filter_dict['cutoff'], window=filter_dict['window'], fs=sampling_frequency)
        filtered = signal.lfilter(taps, 1.0, biosignal)
    return filtered


def resample(df):
    X = df.drop(df.columns[-1], axis = 1)
    y = df.iloc[:,-1]
    
    underSampler = RandomUnderSampler()
    X_sampled, y_sampled = underSampler.fit_resample(X,y)
    
    return pd.concat([X_sampled, y_sampled], axis=1)
