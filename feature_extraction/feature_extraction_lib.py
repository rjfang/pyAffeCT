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
import matplotlib.pyplot as plt
import math
from scipy.stats import iqr
from scipy.fftpack import fft

from pyAffeCT.feature_extraction.pyEDA.main import *

k = 32

def generate_GSR_feature_list():
    FEATURE_LIST = ['MAV','P2P','Peak','RMS','SD','VAR','Range','Intrange','MeanFreq'\
                    ,'MedianFreq','ModeFreq']
    feature_list = []
    
    for feature in FEATURE_LIST:
        feature_list.append('GSR_'+feature)
    
    return feature_list


def generate_GSR_features(data):
    feature_row = []
    data = data.reset_index(drop = True)
    feature_row.append(MAV(data))
    feature_row.append(P2P(data))
    feature_row.append(Peak(data))
    feature_row.append(RMS(data))
     
    feature_row.append(SD(data))
    feature_row.append(VAR(data))
    feature_row.append(Range(data))
    feature_row.append(intrange(data))
     
    feature_row.append(MeanFreq(data))
    feature_row.append(MedianFreq(data))
    feature_row.append(ModeFreq(data))
    
    return feature_row

def generate_pyEDA_feature_list():
    feature_list = []
    for i in range(k):
        feature_list.append('pyEDA_'+str(i))
    return feature_list


def pyEDA_train(data, sampling_rate, window_length, model_name):

    model_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    prepare_automatic(data, sample_rate=sampling_rate, k=32, epochs=10,model_path=os.path.join(model_path, model_name))


def pyEDA_process(data,model_name=None):
    model_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    gsr_feature = process_automatic(gsr_signal=data,model_path=os.path.join(model_path, model_name))
    return gsr_feature

def generate_EMG_feature_list():
    FEATURE_LIST = ['MAV','P2P','Peak','RMS','SD','VAR','Range','Intrange','MeanFreq'\
                    ,'MedianFreq','ModeFreq']
    feature_list = []
    
    for feature in FEATURE_LIST:
        feature_list.append('EMG_'+feature)

    return feature_list

def generate_EMG_features(data):
    feature_row = []
    data = data.reset_index(drop = True)
    feature_row.append(MAV(data))
    feature_row.append(P2P(data))
    feature_row.append(Peak(data))
    feature_row.append(RMS(data))
     
    feature_row.append(SD(data))
    feature_row.append(VAR(data))
    feature_row.append(Range(data))
    feature_row.append(intrange(data))
     
    feature_row.append(MeanFreq(data))
    feature_row.append(MedianFreq(data))
    feature_row.append(ModeFreq(data))
    return feature_row


'''===================    Amplitude    ========================='''
def MAV(data):
    res = list(map(abs,data))
    return sum(res)/len(data)

def P2P(data):
    return float(data[np.argmax(data,axis=0)]) - float(data[np.argmin(data,axis=0)])
        

def Peak(data):
    return max(data)

def RMS(data):
    return math.sqrt(sum([x ** 2 for x in data]) / len(data))



'''===================    Variability    ========================='''
def SD(data):
    return np.std(data)

def VAR(data):
    return np.var(data)

def Range(data):
    return max(data) - min(data)

def intrange(data):
    return iqr(data)


'''===================    Frequency    ========================='''
def MeanFreq(list_signal):
    list_signal = list_signal.values.tolist()
    N = len(list_signal)
    y = list_signal
    yf = fft(y)
    Yf = 2.0/N * np.abs(yf[0:N//2])
    out = np.mean(Yf)
    return out

def MedianFreq(list_signal):
    list_signal = list_signal.values.tolist()
    N = len(list_signal)
    y = list_signal
    yf = fft(y)
    Yf = 2.0/N * np.abs(yf[0:N//2])
    out = np.median(Yf)
    return out

def ModeFreq(list_signal):
    list_signal = list_signal.values.tolist()
    N = len(list_signal)
    y = list_signal
    yf = fft(y)
    Yf = 2.0/N * np.abs(yf[0:N//2])
    out = max(Yf)
    return out
    
def ZeroCrossings(list_signal):
    list_signal = list_signal.values.tolist()
    ACF = np.array(list_signal[:-1])*np.array(list_signal[1:])
    ZC = ACF[ np.where( ACF <0 ) ]
    out = len(ZC)
    return out

'''===================    Entropy    ========================='''
def ApEn(data):
    m = 2
    r = 3
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        x = [[data[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))
    N = len(data)
    return abs(_phi(m + 1) - _phi(m))    


def ShannonEn(data):
    list_signal = np.array(data).tolist()
    if not list_signal:
        return 0
    entropy = 0
    for x in list(set(list_signal)):
        p_x = float(list_signal.count(x))/len(list_signal)
        if p_x > 0:
            entropy += - p_x*math.log(p_x, 2)
    return entropy       


def SampleEn(list_signal):
    def samp_entropy(X, M, R):
        def embed_seq(X,Tau,D):
            	N =len(X)
            	if D * Tau > N:
            		print("Cannot build such a matrix, because D * Tau > N") 
            		exit()
            	if Tau<1:
            		print("Tau has to be at least 1")
            		exit()
            	Y=np.zeros((N - (D - 1) * Tau, D))
            	for i in range(0, N - (D - 1) * Tau):
            		for j in range(0, D):
            			Y[i][j] = X[i + j * Tau]
            	return Y
    
        def in_range(Template, Scroll, Distance):
            	for i in range(0,  len(Template)):
            			if abs(Template[i] - Scroll[i]) > Distance:
            			     return False
            	return True
    
        N = len(X)
        Em = embed_seq(X, 1, M)	
        Emp = embed_seq(X, 1, M + 1)
        Cm, Cmp = np.zeros(N - M - 1) + 1e-100, np.zeros(N - M - 1) + 1e-100
        # in case there is 0 after counting. Log(0) is undefined.
    
        for i in range(0, N - M):
            for j in range(i + 1, N - M): # no self-match
            #			if max(abs(Em[i]-Em[j])) <= R:  # v 0.01_b_r1 
                if in_range(Em[i], Em[j], R):
                    Cm[i] += 1
            #			if max(abs(Emp[i] - Emp[j])) <= R: # v 0.01_b_r1
                    if abs(Emp[i][-1] - Emp[j][-1]) <= R: # check last one
                        Cmp[i] += 1
        Samp_En = math.log(sum(Cm)/sum(Cmp))
        return Samp_En
    return samp_entropy(list_signal,2,3)


def SpectralEn(list_signal):
    list_signal = np.array(list_signal).tolist()
    PSD=abs(fft(list_signal))**2
    # Normalization
    PSD_Norm = (PSD/max(abs(PSD))).tolist()
    # Entropy Calculation
    PSDEntropy = 0
    for x in list(set(PSD_Norm)):
        p_x = float(PSD_Norm.count(x))/len(PSD_Norm)
        if p_x > 0:
            PSDEntropy += - p_x*math.log(p_x, 2)
    return PSDEntropy