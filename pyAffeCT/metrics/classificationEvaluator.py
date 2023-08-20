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
from pyAffeCT.visualization.plot_lib import plot_ROC
from sklearn.metrics import average_precision_score, roc_auc_score, \
    accuracy_score, precision_score, recall_score, f1_score

pd.set_option('display.max_columns', None)
class classificationEvaluator():
    """ a class for classification task results evaluation """
    metrics = ['accuracy','precision','recall','f1','roc_auc','pr_auc']
    def __init__(self, metrics=None):
        if metrics is not None:
            self.metrics = metrics
    
    def evaluate(self, y_true, y_pred=None, num_classes=None, y_prob=None, threshold=0.5, plot_roc=True):
        if num_classes is None:
            num_classes =  len(np.unique(y_true))

       
        # for multi-class classification
        average = 'binary'
        if num_classes > 2 :
            average = 'weighted'
            if 'roc_auc' in self.metrics:
                self.metrics.remove('roc_auc')
            if 'pr_auc' in self.metrics:
                self.metrics.remove('pr_auc') 
            
        
        # if y_prob not provided
        if y_prob is None:
            if 'roc_auc' in self.metrics:
                self.metrics.remove('roc_auc')
            if 'pr_auc' in self.metrics:
                self.metrics.remove('pr_auc') 
        
        if y_pred is None:
            y_pred = pd.DataFrame(columns=y_prob.keys())
            for column in y_pred.columns:
                y_pred[column] = np.argmax(y_prob[column], axis=1)
                
        # initialize a dataframe to store scores
        metrics_df = pd.DataFrame(index=self.metrics)

        if type(y_pred) is pd.core.frame.DataFrame:
            for column in y_pred.columns:
                metrics_list = []
                for metric in self.metrics:
                    if metric == 'accuracy':
                        metrics_list.append(accuracy_score(y_true,y_pred[column]))
                    elif metric == 'precision':
                        metrics_list.append(precision_score(y_true,y_pred[column],average = average))
                    elif metric == 'recall':
                        metrics_list.append(recall_score(y_true,y_pred[column],average = average))
                    elif metric == 'f1':
                        metrics_list.append(f1_score(y_true,y_pred[column],average = average))
        
                    elif metric == 'roc_auc':
                        metrics_list.append(roc_auc_score(y_true,y_prob[column][:,1]))
                        if plot_roc:
                            plot_ROC(y_true,y_prob[column][:,1], title=column)
                    elif metric == 'pr_auc':
                        metrics_list.append(average_precision_score(y_true,y_prob[column][:,1]))
                metrics_df[column] = metrics_list
       
        else:
            metrics_list = []
            for metric in self.metrics:
                if metric == 'accuracy':
                    metrics_list.append(accuracy_score(y_true,y_pred))
                elif metric == 'precision':
                    metrics_list.append(precision_score(y_true,y_pred))
                elif metric == 'recall':
                    metrics_list.append(recall_score(y_true,y_pred))
                elif metric == 'f1':
                    metrics_list.append(f1_score(y_true,y_pred))
    
                elif metric == 'roc_auc':
                    metrics_list.append(roc_auc_score(y_true,y_prob[:,1]))
                    if plot_roc:
                        plot_ROC(y_true,y_prob[column][:,1], title=column)
                elif metric == 'pr_auc':
                    metrics_list.append(average_precision_score(y_true,y_prob[:,1]))
            metrics_df['Results'] = metrics_list
        
        return metrics_df
    
    