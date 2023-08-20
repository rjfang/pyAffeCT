# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 rjfang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from pyAffeCT.models import traditionalClassifier
from pyAffeCT.metrics import classificationEvaluator
from pyAffeCT.datasets import WESAD, BioVidEmo, BioVid, ASCERTAIN, HCI_Tagging




# constructing a dataset instance
#dataset = WESAD('D:/Google Drive/02 Research/01 Affective Computing/03 Benchmark/pyAffeCT/data/02 WESAD/WESAD')
#dataset = BioVidEmo('/content/drive/MyDrive/02 Research/01 Affective Computing/03 Benchmark/pyAffeCT/data/04 BioVidEmo/bio_raw')
#dataset = BioVid('D:/Google Drive/02 Research/01 Affective Computing/03 Benchmark/pyAffeCT/data/03 BioVid')
#dataset = ASCERTAIN('D:/Google Drive/02 Research/01 Affective Computing/03 Benchmark/pyAffeCT/data/05 ASCERTAIN')
dataset = HCI_Tagging('D:/Google Drive/02 Research/01 Affective Computing/03 Benchmark/pyAffeCT/data/06 HCI')

# view the subject id list
subject_list = dataset.subject_list
modality_included = ['ECG1','GSR','RSEP']
window_length = 5
num_classes = 2
task = 'Arousal'

from pyAffeCT.models import traditionalClassifier
from pyAffeCT.metrics import classificationEvaluator


# prepare for the dataframe, set number of classess to be 2
df_train, df_test = dataset.prepare_dataframe(subject_included=subject_list[:], split_by_subject=True, window_length = window_length, modality_included=modality_included, label_task = task, num_classes=2)
df_train.head()

X_train = df_train.loc[:,df_train.columns!='Label']
y_train = df_train['Label']
X_test = df_test.loc[:,df_test.columns!='Label']
y_test = df_test['Label']

# instantiate tranditional classifier
clf = traditionalClassifier()

# fit to the training set, tuning to the best hyperparameters
clf.tune_hyperparams(X_train,y_train)

# predict to classess and probabilities
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# instantiate evaluator and evaluate results
evaluator = classificationEvaluator()
results = evaluator.evaluate(y_test, y_prob=y_prob, plot_roc=False)
print(results)


Following_test = False
if Following_test:
    
    from pyAffeCT.feature_extraction.feature_extraction_lib import generate_GSR_feature_list

    remove_list = generate_GSR_feature_list()
    
    X_train = df_train.loc[:,remove_list]
    y_train = df_train['Label']
    X_test = df_test.loc[:,remove_list]
    y_test = df_test['Label']

    # instantiate tranditional classifier
    clf = traditionalClassifier()

    # fit to the training set, tuning to the best hyperparameters
    clf.tune_hyperparams(X_train,y_train)

    # predict to classess and probabilities
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # instantiate evaluator and evaluate results
    evaluator = classificationEvaluator()
    results = evaluator.evaluate(y_test, y_prob=y_prob, plot_roc=False)
    print(results)
    
    
    