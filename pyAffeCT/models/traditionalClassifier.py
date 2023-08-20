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
from math import sqrt

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier


from sklearn.model_selection import RandomizedSearchCV

class traditionalClassifier():
    """
    The class that combines multiple traditional classifiers
    """
    classifiers = dict()
    
    def __init__(self, classifiers_included = ['SVM','GNB','BNB','MNB','DT','RF','KNN','Extra Tree','Bagging', 'AdaBoost','XGBoost'], using_tpot=False, random_state=42):
        """
        Initialization function

        Parameters
        ----------
        classifiers_included : list, optional
            classifiers to be included, select within the default list. The default is ['SVM','DT','RF'].
        random_state : int, optional
            random state. The default is 42.

        Returns
        -------
        None.

        """
        
        for clf_name in classifiers_included:
            if clf_name == 'SVM':
                self.classifiers[clf_name] = svm.SVC(kernel='rbf',random_state=random_state,probability=True)
            elif clf_name == 'GNB':
                self.classifiers[clf_name] = GaussianNB()
            elif clf_name == 'BNB':
                self.classifiers[clf_name] = BernoulliNB()
            elif clf_name == 'MNB':
                self.classifiers[clf_name] = MultinomialNB()
            elif clf_name == 'DT':
                self.classifiers[clf_name] = DecisionTreeClassifier(random_state=random_state)
            elif clf_name == 'RF':
                self.classifiers[clf_name] = RandomForestClassifier(random_state=random_state)
            elif clf_name =='KNN':
                self.classifiers[clf_name] = KNeighborsClassifier()
            elif clf_name == 'Extra Tree':
                self.classifiers[clf_name] = ExtraTreesClassifier(random_state=random_state)
            elif clf_name == 'Bagging':
                self.classifiers[clf_name] = BaggingClassifier(random_state=random_state)
            elif clf_name == 'AdaBoost':
                self.classifiers[clf_name] = AdaBoostClassifier(random_state=random_state)
            elif clf_name == 'XGBoost':
                self.classifiers[clf_name] = XGBClassifier(random_state=random_state)
        
    
    def add_classifier(self, new_classifier, new_classifier_name='New Classifier'):
        """
        function:
            add a new classifier to the list
        """
        self.classifiers[new_classifier_name] = new_classifier
    
    def tune_hyperparams(self, X, y):
        """
        Tune hyper-parameters for each classifiers with a set of given X and y

        Parameters
        ----------
        X : dataframe/ndarray
            a set of data to tune the hyperparameter.
        y : list/series
            corresponding y.

        Returns
        -------
        None.

        """
        
        grid_search_params = {
            'SVM':{
                'C': np.linspace(0.1,1,9),
                'gamma': np.linspace(0.01,0.5,20)
                },
            'GNB':{
                'var_smoothing': [1e-9, 1e-7, 1e-5],
                },
            'BNB':{
                'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                },
            'MNB':{
                'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                },
            'RF':{
                'n_estimators':list(range(10,200,10)),
                'criterion':['gini','entropy','log_loss'],
                'max_depth':list(range(10,100,10))+[None],
                'min_samples_split':list(range(2,10,2)),
                'max_features':list(range(1,int(sqrt(X.shape[1])),3))
                },
            'DT':{
                'max_depth':list(range(10,100,10))
                },
            'Extra Tree':{
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                },
            'KNN':{
                'n_neighbors':list(range(1,20,2))
                },
            'XGBoost':{
                "n_estimators": [20,50,100],
                "max_depth": range(3, 10),
                "learning_rate": [1e-2, 1e-1, 0.5, 1.],
                "subsample": np.arange(0.05, 1.01, 0.05),
                "min_child_weight": range(1, 21),
                "alpha": [1, 10],
                },

            }
        
        for clf_name, clf in self.classifiers.items():
            if clf_name in grid_search_params.keys():
                searcher = RandomizedSearchCV(clf, grid_search_params[clf_name], n_iter=5,random_state=42)
                searcher.fit(X,y)
                self.classifiers[clf_name] = searcher.best_estimator_
            else:
                self.classifiers[clf_name].fit(X,y)

        return
    
    
    def fit(self, X, y):
        """
        Fit all classifiers to the data
        """
        for clf in self.classifiers.values():
            clf.fit(X,y)
        
            
    
    def predict(self, X):
        """
        To predict a series of new X with all classifiers
        """
        y_pred = pd.DataFrame(columns=self.classifiers.keys())
        for i, clf in enumerate(self.classifiers.values()):
            y_pred.iloc[:,i] = clf.predict(X)

        return y_pred
        
    
    def predict_proba(self, X):
        """
        To predict new X and return probability

        Parameters
        ----------
        X : an array or new X

        Returns
        -------
        None.

        """

        y_prob = {}
        for clf_name, clf in self.classifiers.items():
            y_prob[clf_name] = clf.predict_proba(X)
        return y_prob
    
