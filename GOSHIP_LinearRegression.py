#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:44:24 2021

@author: Ellen
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

GitHubDir='/Users/Ellen/Documents/GitHub/6862-project/'
InputDataDir=GitHubDir+'data/GOSHIP_Data/GOSHIP_QC_FeatureEncodedData.csv'
FigDir=GitHubDir+'figures/LinearRegression/'

Data=pd.read_csv(InputDataDir)
Data=Data.drop(['Unnamed: 0'], axis = 1)

# # Update month column
# new_m=[[]]*Data.loc[:,'MONTH'].shape[0]

# for i in np.arange(Data.loc[:,'MONTH'].shape[0]):
#     tt=Data.loc[:,'MONTH'][0][1:-1].split(' ')
    
#     # n_tt=np.zeros(len(tt))
#     # n_tt[:]=np.NaN
    
#     n_tt=[int(float(j)) for j in tt]
#     n_tt=np.array(n_tt)
#     # for j in np.arange(len(tt)):
#     #     n_tt[j]=int(float(tt[j]))
    
#     new_m[i]=n_tt

# Data['MONTH']=new_m

X=Data.iloc[:,:-2]
Y1=Data.loc[:,'PHSP'].to_numpy()
Y2=Data.loc[:,'SILI']

## Phosphate ##
P_reg=LinearRegression().fit(X,Y1)
P_score=P_reg.score(X,Y1)
P_coeff=P_reg.coef_
P_inter=P_reg.intercept_
P_predict=P_reg.predict(X)

print('\n%% MLR for P %%')
print('Score: ', P_score)
print('P coeffs: ', P_coeff)
print('P intercept: ', P_inter)

p_x1 = np.arange(-6,3)

plt.figure()
plt.scatter(Y1, P_predict)
plt.plot(p_x1, p_x1, 'k-')
plt.xlabel('P true')
plt.ylabel('P predicted')
plt.savefig(FigDir+'P_LinearRegression.jpg')

plt.figure()
plt.scatter(Y1, P_predict)
plt.plot(p_x1, p_x1, 'k-')
plt.ylim((-6,3))
plt.xlabel('P true')
plt.ylabel('P predicted')
plt.savefig(FigDir+'P_LinearRegression_Cropped.jpg')

fig, axs = plt.subplots (2,4, figsize=(10,6))
Features=['MONTH','LATITUDE', 'LONGITUDE','PRES','TEMP','SAL','OXY','NITR','PHSP','SILI']
for i in np.arange(1,8):
    
    ax = plt.subplot(2,4,i)
    x_data=X.iloc[:, 11+i]
    
    ax.scatter(x_data, P_predict)
    ax.set_ylabel('P predicted')
    ax.set_xlabel(Features[i])
    ax.set_title(Features[i])

plt.subplots_adjust(hspace=0.4, wspace=0.5)
plt.suptitle('P predicted vs. x-values')
plt.savefig(FigDir+'P_LinearRegression_Xvalues.jpg')

## Silicte ##
S_reg=LinearRegression().fit(X,Y2)
S_score=S_reg.score(X,Y2)
S_coeff=S_reg.coef_
S_inter=S_reg.intercept_
S_predict=S_reg.predict(X)

print('\n%% MLR for S %%')
print('Score: ', S_score)
print('P coeffs: ', S_coeff)
print('P intercept: ', S_inter)

s_x1 = np.arange(-6,4)

plt.figure()
plt.scatter(Y1, S_predict)
plt.plot(p_x1, p_x1, 'k-')
plt.xlabel('S true')
plt.ylabel('S predicted')
plt.savefig(FigDir+'S_LinearRegression.jpg')

plt.figure()
plt.scatter(Y1, S_predict)
plt.plot(p_x1, p_x1, 'k-')
plt.ylim((-6,4))
plt.xlabel('S true')
plt.ylabel('S predicted')
plt.savefig(FigDir+'S_LinearRegression_Cropped.jpg')

fig, axs = plt.subplots (2,4, figsize=(10,6))
for i in np.arange(1,8):

    ax = plt.subplot(2,4,i)
    x_data=X.iloc[:, 11+i]
    
    ax.scatter(x_data, S_predict)
    ax.set_ylabel('S predicted')
    ax.set_xlabel(Features[i])
    ax.set_title(Features[i])

plt.subplots_adjust(hspace=0.4, wspace=0.5)
plt.suptitle('S predicted vs. x-values')
plt.savefig(FigDir+'S_LinearRegression_Xvalues.jpg')

plt.show()