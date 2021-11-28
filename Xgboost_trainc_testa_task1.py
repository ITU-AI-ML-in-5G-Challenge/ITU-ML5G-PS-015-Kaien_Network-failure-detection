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
#import joblib
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
import shap


start_time = time.time()
print("------------------------------ 1: Loading Data from csv------")

fx=open('csvFile/dataset_abc.csv','rb')
df = pd.read_csv(fx,index_col=None) 
fx.close
print('df:',df.shape)

#print(df.isnull().sum().sort_values(ascending=False))
#df.isnull().sum(axis=0).plot.barh()
#plt.title('Ratio of missing values per columns')
#plt.show()

train = df[(df['traintest']==0) & (df['abc']=='c')]  #a,c
test = df[(df['traintest']==1) & (df['abc']=='a')] #a,c

y_train = train['labelnew'].values  
train = train.drop(['labelnew','abc','traintest','datetime','node'
                    ],axis=1)

print('train:',train.shape)
X_train =  train.values  

y_test = test['labelnew'].values
test = test.drop(['labelnew','abc','traintest','datetime','node'
                    ],axis=1)
X_test =  test.values  

print('X_train', X_train.shape)
print('X_test', X_test.shape)

print("----------------------- Xgboost start ---------------------")
start_time = time.time()

clf = XGBClassifier(
    silent=0,  # 
    nthread = -1, # 
    tree_method= 'exact', #'approx', #'hist', #'approx', #approx,exact,hist
    #max_bin=12, #240, #default=256
    booster='gbtree',#dark  gbtree  
    #booster='dark',#dark
    #sample_type='weighted', #uniform
    #normalize_type='forest', #tree
    #rate_drop=0.9,   
    n_estimators=80,  # 
    max_depth=4, #
    learning_rate=0.2, #
    #min_child_weight = 0.9, default=1
    max_delta_step=1, #
    #scale_pos_weight = 130, # 1778034/986# 
    #base_score=0.90, #
    #reg_lambda=10, #default=1
    #gamma=0,
    subsample=0.85, #
    #gamma = 0.1,# 0.1~0.2
    #subsample=1, # 
    colsample_bytree=0.8, # 
    #reg_lambda=1, #
    #reg_alpha=0, # L1
    #max_delta_step=2,#1  # 
    #scale_pos_weight = 0.0017499，#sum(negative instances)/sum(positive instances)
    #booster='gbtree',#dark
    objective ='binary:logistic',#'binary:logistic', #'multi:softprob', #'multi:softmax' 
    #base_score=0.66,worse
    #bjective ='multi:softprob',#'binary:logistic', #'multi:softprob', #'multi:softmax' 
    #um_class = 2,  # multisoftmax
    #seed = 1440,  # 
    #missing=None,
    #eval_metric ='auc',#error
    )

print('make model')
clf.fit(X_train, y_train)
print('save model')
joblib.dump(clf, 'module/Xgboost.pkl')

print("Xgboost training cost time : ",time.time()-start_time)

print('# -----------------6: Xgboost: train scoring ---------------')
#load model
rfcbuild = joblib.load('module/Xgboost.pkl')
TrainResult = rfcbuild.predict(X_train)

colNum = TrainResult.shape[0]
TrainResult = TrainResult.reshape(colNum,1)

print("confusion_matrix:")
print(confusion_matrix(y_train,TrainResult))
print('---------')
print(classification_report(y_train,TrainResult))
roc_auc1 = roc_auc_score(y_train,TrainResult)
print("Area under the ROC curve : %f" % roc_auc1)

print("------------------ Xgboost: test scoring ---------------")

TestResult = rfcbuild.predict(X_test)
colNum = TestResult.shape[0]
TestResult = TestResult.reshape(colNum,1)

print("confusion_matrix:")
print(confusion_matrix(y_test,TestResult))
print('---------')
print(classification_report(y_test,TestResult))
roc_auc1 = roc_auc_score(y_test,TestResult)
print("Area under the ROC curve : %f" % roc_auc1)

# Feature Importance
plt.figure(figsize=(10, 5))
fea = rfcbuild.feature_importances_
maxrange=np.max(fea)-np.min(fea)

fea_normed = 100 * fea/np.max(fea)
#fea_normed = fea/maxrange
#fea_normed = (fea-np.min(fea))/maxrange
#print('Importance score: ',type(fea_normed),fea_normed.shape,fea_normed)
np.set_printoptions(suppress=True)
print('Importance score: ')
print(np.round(fea_normed,2))

plt.figure(figsize=(10, 5))
plt.bar(range(len(rfcbuild.feature_importances_)), fea_normed)
plt.title("Xgboost: Feature Importance")
plt.xlabel("Features")
plt.ylabel("Normalized Importance Score")
plt.show()  

print(' ---------  Output Test (predict) results to csv  -----')
print('----------- Xgboost train and testsuccessful completed-----------------')

