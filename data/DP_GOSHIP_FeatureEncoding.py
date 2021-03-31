#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:14:00 2021

@author: Ellen
"""

import pandas as pd
import numpy as np

# Feature encoding
GitHubDir='/Users/Ellen/Documents/GitHub/6862-project/'
GOSHIPData=pd.read_csv(GitHubDir+'data/GOSHIP_Data/QCFilteredData.csv')
OutDir=GitHubDir+'data/GOSHIP_Data/'

Features=['MONTH','LATITUDE', 'LONGITUDE','PRES','TEMP','SAL','OXY','NITR','PHSP','SILI']

FeatEncode=['M','R','R','S','S','S','S','S','S','S']

# Processing options
# Raw 'R'
# Standardized 'S'
# Month encoding 'M'

for i in np.arange(len(Features)):
    
    # Encode each feature properly according to flag
    feat_flag = FeatEncode[i]
    
    if feat_flag == 'R':
        EncodedFeatures=GOSHIPData.loc[:,Features[i]]
    elif feat_flag == 'S':
        non_data=GOSHIPData.loc[:,Features[i]]
        mean_data=non_data.mean()
        std_data=non_data.std()
        EncodedFeatures=(non_data-mean_data)/std_data
    elif feat_flag == 'M':
        non_data=GOSHIPData.loc[:,Features[i]].to_numpy()
        temp_data=np.zeros((len(non_data), 12))
        
        for m in np.arange(len(temp_data)):
            m_ind=non_data[m]     
            temp_data[m,:m_ind]=1
        
        EncodedFeatures=pd.DataFrame(temp_data, columns=['M1','M2','M3', 'M4', 'M5','M6', 'M7', 'M8', 'M9','M10','M11', 'M12'])
    
    # Save encoded features in a new data frame 
    
    if i == 0:
        EncodedData=EncodedFeatures
    else:
        EncodedData[Features[i]]=EncodedFeatures
        
# Save Data
EncodedData.to_csv(OutDir+'GOSHIP_QC_FeatureEncodedData.csv')    