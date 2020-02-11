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

data_set = 'gene'  #@param {type: "string"}
root_path = '../../datasets/'
#cleaned data without non-attack values
X = pd.read_csv(root_path+data_set+'.csv',  usecols=[i for i in range(1, 20532)])
Y = pd.read_csv(root_path+data_set+'_label.csv', usecols=[1])

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

# cluster and score
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import adjusted_rand_score

score = []
for i in range(len(X.columns)): # loop number of features
    K = MiniBatchKMeans(n_clusters=len(Y[Y.columns[0]].unique()), random_state=0)
    pred = K.fit_predict(X.iloc[:, [i]].values)
    s = adjusted_rand_score(Y[Y.columns[0]].values,pred)
    score.append(s)


# Rank the features and sort

s2 = score
np.asarray(s2)

s1 = []
for i in range(len(X.columns)):
    s1.append(i)

np.asarray(s1)

li = list(zip(s1, s2))

def sortSecond(val): 
    return val[1] 

li.sort(key = sortSecond, reverse=True) 

# Create a copy of X dataframe with columns sorted by score

titles = []

for i in range(len(X.columns)):
    p = X.columns[li[i][0]]
    titles.append(p)


X1 = pd.DataFrame(columns=titles)

for i in range(len(X.columns)):
    X1[X1.columns[i]] = X[X.columns[li[i][0]]]




# Recursive Feature Elemination from # of features to 0 and keep the accuracy score of each

accuracy = []
X2 = X1.copy()

# for i in range(len(X1.columns)-1,-1,-1):
for i in range(len(X1.columns)-1): 
    # remove lowest scored column
    X1.drop(X1.columns[len(X1.columns)-1], axis=1, inplace=True)
    # begin cross-validation
    cv_accuracy =[]
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, random_state=0)
    for train, test in kf.split(X1,Y):
        # classifyer
        r = rf(random_state=0, n_jobs=-1) 
        # train test split
        x_train = X1.iloc[train]
        x_test = X1.iloc[test]
        y_train = Y.iloc[train]
        y_test = Y.iloc[test]
        r.fit(x_train, y_train.values.ravel())
        y_pred = r.predict(x_test)
        cv_accuracy.append(accuracy_score(y_test, y_pred)) 
    accuracy.append(np.mean(cv_accuracy))

# best score calcuation
index = accuracy.index(max(accuracy))
X = X2.iloc[:,0:len(X.columns)-index]


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