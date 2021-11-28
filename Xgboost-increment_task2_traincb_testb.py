#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov.28, 2021
@author: Kaien ABE @Kobe Univ.
"""

from numpy import *
from sklearn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns 
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from graphviz import Digraph
from sklearn.datasets import make_gaussian_quantiles
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split, BaseCrossValidator
import time,datetime
import sqlite3
import pandas.io.sql as psql
import csv,os,imp
from sklearn.utils import class_weight
from sklearn import manifold, datasets
from sklearn.preprocessing import normalize,Normalizer,StandardScaler,MinMaxScaler,scale 
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error as mse

start_time = time.time()
print("------------------------------ 1: Loading Data from csv------")

fx=open('csvFile/dataset_abc.csv','rb')
df = pd.read_csv(fx,index_col=None) 
fx.close

train = df[(df['traintest']==0) & (df['abc']=='c')]  #if borrow task_1 model a
test = df[(df['traintest']==1) & (df['abc']=='b')] #a,c

y_train = train['labelnew']
train = train.drop(['labelnew','abc','traintest','datetime','node'
                    ],axis=1)
X_train =  train 

y_test = test['labelnew']
test = test.drop(['labelnew','abc','traintest','datetime','node'
                    ],axis=1)
X_test =  xgb.DMatrix(test, label=y_test) 

print('X_train', X_train.shape)
print('test', test.shape)

print('------------------------- 2: training base model in task_1 --')
params = {
         'silent':0,
         'nthread':-1,
         'tree_method':'hist',
         'booster':'gbtree',
         'max_depth':4,
         'learning_rate':0.1,
         'objective':'binary:logistic',
        }

dtrain = xgb.DMatrix(X_train, label=y_train)
clf0 = xgb.train(params, dtrain, num_boost_round=60)
clf0.dump_model('module/clf.model')
#y_pred = clf.predict()
print('------------------------- 2: increment model based task_1 model --')
params = {
         'silent':0,
         'nthread':-1,
         'tree_method':'hist',
         'booster':'gbtree',
         'max_depth':4,
         'learning_rate':0.1,
         'objective':'binary:logistic',
        }

train = df[(df['traintest']==0) & (df['abc']=='b')]
y_train = train['label']
train = train.drop(['label','labelnew','label2','abc','traintest','datetime','name'#,'operstatus1onehot','operstatus2onehot','operstatus3onehot','minutestatusconditiononehot','minutestatusconditiononehot','memorystatusonehot'
                    ],axis=1)
X_train =  train#.values  

dtrain = xgb.DMatrix(X_train, label=y_train)
clfb = xgb.train(params, dtrain, num_boost_round=30, xgb_model=clf0)
clfb.dump_model('module/clfb.model')
y_pred = clfb.predict(X_test)
print('mse:', mse(clfb.predict(X_test), y_test))
cm = confusion_matrix(y_test, (y_pred>0.5))
print(cm)
cr = classification_report(y_test,(y_pred>0.5))
print(cr)

print("----------------------- 5: Xgboost start ---------------------")

