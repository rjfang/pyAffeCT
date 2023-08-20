# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 rjfang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys 
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

colorMap = {0: (102,194,165),
            1: (252,141,98),
            2: (141,160,203),
            3: (231,138,195),
            4: (166,216,84),
            5: (255,217,47),
            6: (229,196,148)}

            
def ecg_plot_sub(timestamp, ecg, label = None, sampling_rate = 1000, xrange='5s'):
    # downsample if signal too long
    if len(timestamp) > 50000:
        if label is not None:
            df = pd.DataFrame({'TIMESTAMP':timestamp, 'ECG':ecg, 'Label':label}) 
        else:
            df = pd.DataFrame({'TIMESTAMP':timestamp, 'ECG':ecg})
        df = df.resample('20ms',on='TIMESTAMP').first()
        sampling_rate = 50
        if len(df) > 50000:
            df = df.iloc[:50000]
        df.reset_index(drop=True, inplace=True)
        ecg = df['ECG']
        timestamp = df['TIMESTAMP']
        if 'Label' in df.columns:
            label = df['Label']

        
    ecg = ecg[:len(timestamp)]
    
    ecg_peaks = nk.ecg_peaks(ecg, sampling_rate = sampling_rate)[0]['ECG_R_Peaks']
    ecg_peaks.index = ecg.index
    ecg_peaks = ecg[ecg_peaks==1]
    
    traceECG = go.Scatter(
        x = timestamp,
        y = ecg,
        mode = 'lines',
        name = 'ECG',
        marker = dict(size=5,color='rgba(16,112,2,0.8)'),
        line=dict(width=3),
        )
    
    traceECGpeaks = go.Scatter(
        x = timestamp.loc[ecg_peaks.index],
        y = ecg_peaks,
        mode = 'markers',
        name = 'ECG Peaks',
        marker = dict(size=10,color='rgba(255,0,0,0.8)'),
        )
    
    if label is not None:
        label = label[:len(timestamp)]
        traceLabel = go.Scatter(
            x = timestamp,
            y = label,
            mode = 'lines',
            name = 'Label',
            marker = dict(size=5, color='rgba(80,26,80,0.8)'),
            line=dict(width=3),
            yaxis='y2'
            )
        data = [traceECG, traceECGpeaks, traceLabel]
    
    else:
        data = [traceECG, traceECGpeaks]
    
    layout = dict(title = 'ECG Plot',
                  xaxis=dict(title='timestamp',range=[timestamp.iloc[0],timestamp.iloc[0]+pd.Timedelta(xrange)], ticklen=5, zeroline=False),
                  yaxis2=dict(anchor='x',overlaying='y',side='right'))
    
    fig = dict(data=data, layout=layout)
    iplot(fig)



def gsr_plot_sub(timestamp, gsr, label =None, sampling_rate = 1000, xrange='5m'):
    # downsample if signal too long
    if len(timestamp) > 100000:
        if label is not None:
            df = pd.DataFrame({'TIMESTAMP':timestamp, 'GSR':gsr, 'Label':label})
        else:
            df = pd.DataFrame({'TIMESTAMP':timestamp, 'GSR':gsr})
        df = df.resample('1s',on='TIMESTAMP').first()
        sampling_rate = 1
        if len(df) > 100000:
            df = df.iloc[:100000]
        df.reset_index(drop=True, inplace=True)
        gsr = df['GSR']
        timestamp = df['TIMESTAMP']
        if 'Label' in df.columns:
            label = df['Label']
        
        
    gsr = gsr[:len(timestamp)]
    
    traceGSR = go.Scatter(
        x = timestamp,
        y = gsr,
        mode = 'lines',
        name = 'GSR',
        marker = dict(size=5,color='rgba(16,112,2,0.8)'),
        line=dict(width=3),
        )

    if 'Label' in df.columns:
        label = label[:len(timestamp)]
        traceLabel = go.Scatter(
            x = timestamp,
            y = label,
            mode = 'lines',
            name = 'Label',
            marker = dict(size=5, color='rgba(80,26,80,0.8)'),
            line=dict(width=3),
            yaxis='y2'
            )
        data = [traceGSR, traceLabel]
    else:
        data = [traceGSR]
    
  
    
    layout = dict(title = 'GSR Plot',
                  xaxis=dict(title='timestamp',range=[timestamp.iloc[0],timestamp.iloc[0]+pd.Timedelta(xrange)], ticklen=5, zeroline=False),
                  yaxis2=dict(anchor='x',overlaying='y',side='right'))
    
    fig = dict(data=data, layout=layout)
    iplot(fig)



def filter_plot_sub(timestamp, df_original, df_filtered, filter_dict, title_list, xrange):       
    fig = make_subplots(rows = len(title_list),
                        cols=1,
                        subplot_titles=title_list,
                        x_title='Time',
                        shared_xaxes  = True
                        )
    i = 1
    for column in filter_dict.keys():
        fig.append_trace(go.Scatter(
            x = timestamp,
            y = df_original[column],
            mode = 'lines',
            name = df_original[column].name,
            marker = dict(size=5),
            line=dict(width=3),
            ), i, 1
            )
        i=i+1
        fig.append_trace(go.Scatter(
            x = timestamp,
            y = df_filtered[column],
            mode = 'lines',
            name = df_filtered[column].name+'_filtered',
            marker = dict(size=5),
            line=dict(width=3),
            ), i, 1
            )
        i=i+1
        
    fig.update_xaxes(dict(range=[timestamp.iloc[0],timestamp.iloc[0]+pd.Timedelta(xrange)]),row=1, col=1)

    iplot(fig)
    

def feature_plot_sub(timestamp, feature_series, label=None, xrange='10+m'):
    feature_series = feature_series[:len(timestamp)]
    
    # case if only one column given
    if type(feature_series) == pd.core.series.Series:
        traceFeature = go.Scatter(
            x = timestamp,
            y = feature_series,
            mode = 'lines',
            name = feature_series.name,
            marker = dict(size=5),
            line=dict(width=3),
            )
        data = [traceFeature]
        
        if label is not None:
            label = label[:len(timestamp)]
            traceLabel = go.Scatter(
                x = timestamp,
                y = label,
                mode = 'lines',
                name = label.name,
                marker = dict(size=5, color='rgba(80,26,80,0.8)'),
                line=dict(width=3),
                yaxis='y2'
                )
            data.append(traceLabel)
        
            
        layout = dict(title = 'GSR Plot',
                      xaxis=dict(title='timestamp',range=[timestamp.iloc[0],timestamp.iloc[0]+pd.Timedelta(xrange)], ticklen=5, zeroline=False),
                      yaxis2=dict(anchor='x',overlaying='y',side='right'))
        
        fig = dict(data=data, layout=layout)
        iplot(fig)
    
    # case if two or more columns to plot
    elif len(feature_series.columns) >= 2:
        fig = make_subplots(rows = len(feature_series.columns)+1,
                            cols=1,
                            subplot_titles=feature_series.columns.values.tolist().append('Label'),
                            x_title='Time',
                            shared_xaxes  = True
                            )
        i=1
        for column in feature_series.columns:
            fig.append_trace(go.Scatter(
                x = timestamp,
                y = feature_series[column],
                mode = 'lines',
                name = feature_series[column].name,
                marker = dict(size=5),
                line=dict(width=3),
                ), i, 1
                )
            #fig.update_xaxes(dict(range=[timestamp.iloc[0],timestamp.iloc[0]+pd.Timedelta(xrange)]),row=i, col=1)
            i=i+1
        if label is not None:
            label = label[:len(timestamp)]
            fig.append_trace(go.Scatter(
                x = timestamp,
                y = label,
                mode = 'lines',
                name = label.name,
                marker = dict(size=5,color='rgba(16,112,2,0.8)'),
                line=dict(width=3),
                ), i, 1
                ) 
        
        fig.update_xaxes(dict(range=[timestamp.iloc[0],timestamp.iloc[0]+pd.Timedelta(xrange)]),row=i, col=1)
        
            
        
        
        iplot(fig)
        
        
    else:
        raise Exception('Invalid Arguments.')

    

def heat_map(df_feature):
    plt.figure(figsize=(15,13))
    sns.heatmap(df_feature.corr(),annot=True,fmt='.2f',annot_kws={'size':8,'weight':'bold'},cmap='rainbow')
    
    


def plot_ROC(y,prob, title=' '):
    fpr,tpr,threshold = roc_curve(y,prob) ###
    roc_auc = auc(fpr,tpr) ###
 
    plt.figure()
    lw = 2
    plt.figure(figsize=(4,4), dpi=100)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) #
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example '+title)
    plt.legend(loc="lower right")
 
    plt.show()


