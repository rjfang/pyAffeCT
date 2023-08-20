# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 rjfang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from pyAffeCT.datasets import BioVidEmo
from pyAffeCT.models import traditionalClassifier
from pyAffeCT.metrics import classificationEvaluator


# constructing a dataset instance
biovidemo = BioVidEmo('D:\\Benchmark_data/04 BioVidEmo/bio_raw')

# view the subject id list
subject_list = biovidemo.subject_list

# prepare for the dataframe, set number of classess to be 3
df_train, df_test = biovidemo.prepare_dataframe(subject_included=subject_list[:3], \
                                                split_by_subject=False, num_classes=5, isResample=False)
df_train.head()

X_train = df_train.loc[:,df_train.columns!='Label']
y_train = df_train['Label']
X_test = df_test.loc[:,df_test.columns!='Label']
y_test = df_test['Label']

# instantiate tranditional classifier
clf = traditionalClassifier()

# fit to the training set
clf.fit(X_train,y_train)

# predict to classess and probabilities
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# instantiate evaluator and evaluate results
evaluator = classificationEvaluator()
results = evaluator.evaluate(y_test, y_prob=y_prob)
print(results)
