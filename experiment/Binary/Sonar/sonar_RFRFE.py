# -*- coding: utf-8 -*-
"""Talking_None.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DLqJwCeAOyqh61a3WfXPMwIXSem1U_W3

# Importing the Data
"""

import sys
print("File: ",sys.argv[0])

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import classification_report, accuracy_score

# from google.colab import drive

# drive.mount('/content/gdrive')
# root_path = 'gdrive/My Drive/Research/Benchmarks/'  #change dir to your project folder

data_set = 'sonar'  #@param {type: "string"}
root_path = '../../datasets/'
#cleaned data without non-attack values
X = pd.read_csv(root_path+data_set+'.csv',  usecols=[i for i in range(60)])
Y = pd.read_csv(root_path+data_set+'.csv', usecols=[60])

from sklearn.preprocessing import StandardScaler
X=X.astype('float')
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# Data Balancing
# from collections import Counter
# print('Original dataset shape %s' % Counter(Y))
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X, Y = sm.fit_resample(X, Y.values.ravel())
# print('Resampled dataset shape %s' % Counter(Y))
X, Y = pd.DataFrame(X), pd.DataFrame(Y)

"""# Feature Engineering"""

# RF RFE includes cross validation to choose the best number of features.

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.feature_selection import RFECV
rfc = rf(n_jobs=-1,random_state=0)
rfe = RFECV(rfc)
X1,Y1 = X.copy(), Y.copy()
X1 = rfe.fit_transform(X1, Y1.values.ravel())
X = pd.DataFrame(X1)


"""# Performance Analysis"""

# must have X and Y
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# handle multiclass classification
def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
  lb = LabelBinarizer()
  lb.fit(y_test)
  y_test = lb.transform(y_test)
  y_pred = lb.transform(y_pred)
  return roc_auc_score(y_test, y_pred, average=average)

# define metrics
accuracy = []
precision = []
recall =[]
f1=[]
roc_auc=[]

# begin cross-validation
kf = StratifiedKFold(n_splits=5, random_state=0)
for train, test in kf.split(X,Y):
  # classifyer
  r = rf(random_state=0, n_jobs=-1, n_estimators=10) 
  # train test split
  X1 = X.iloc[train]
  X2 = X.iloc[test]
  Y1 = Y.iloc[train]
  Y2 = Y.iloc[test]
  # fit
  r.fit(X1,Y1.values.ravel())
  # predict
  Y_pred = r.predict(X2)
  Y_pred = pd.DataFrame(Y_pred)

  # metrics
  accuracy.append(accuracy_score(Y2, Y_pred))
  f1.append(f1_score(Y2, Y_pred, average="weighted"))
  precision.append(precision_score(Y2, Y_pred, average="weighted"))
  recall.append(recall_score(Y2, Y_pred, average="weighted"))
  roc_auc.append(multiclass_roc_auc_score(Y2, Y_pred, average="weighted"))

# print averages
print("Average Accuracy: ",np.mean(accuracy))
print("Average Precision: ",np.mean(precision))
print("Average Recall: ",np.mean(recall))
print("Average F1: ",np.mean(f1))
print("Average ROC_AUC: ", np.mean(roc_auc))
print("Features Selected ", X.shape[1])