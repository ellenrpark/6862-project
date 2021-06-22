#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:58:36 2021
​
@author: Ellen
"""

from joblib import load
from tensorflow import keras
import numpy as np
import pandas as pd
from pyproj import Transformer
import matplotlib.pyplot as plt
#import cartopy as ct
#import cartopy.crs as ccrs
#%%
# load in trained model ('linear_model_phsp','linear_model_sili','dnn_model_phsp','dnn_model_sili') 
def ModelLoader(ModelDir):
    # load in trained model ('linear_model_phsp','linear_model_sili','dnn_model_phsp','dnn_model_sili') 
    ml_model = keras.models.load_model(ModelDir) 
    return ml_model

# load in standard scaler
scaler = load('scaler.bin')

# load in data
data_file = '/Users/nsaoki/Desktop/MIT:WHOI/Classes/Spring 2021/6.862_6.036_ Machine_Learning/6862-project/CroppedESM_InRangeM.csv' ### change this
data = pd.read_csv(data_file, header=0, usecols=[1,2,3,4,5,6,7,8,9])

# project coordinates to cartesian (i.e. easting, northing)
transformer = Transformer.from_crs("epsg:4326", "epsg:3031")

def transform_coords(x):
    return transformer.transform(x[0],x[1])

lat=data.loc[:,'lat']
lon=data.loc[:,'lon']
tmp = data[["lat","lon"]].apply(transform_coords, axis=1)
data['easting'] = tmp.apply(lambda x: x[0]/1e6)
data['northing'] = tmp.apply(lambda x: x[1]/1e6)
data.drop(['lat','lon'], inplace=True, axis='columns') # remove latitude and longitude

lat=data.loc[:,'easting']
lon=data.loc[:,'northing']

# # plot locations of data
# plt.scatter(data['easting'], data['northing'])
# plt.show() # plot

# separate out coordinates from rest of features
# separate out labels (phsp, sili)
test_features = data.copy()
test_coords = test_features[['easting','northing']]
test_features.drop(['easting','northing'], inplace=True, axis='columns')
test_phsp = test_features.pop('phsp')
test_sili = test_features.pop('sili')

# scale features
# note we do not scale easting and northing
scaled_test_features = scaler.transform(test_features)
testing_features = np.concatenate([test_coords, scaled_test_features],axis=1)

# predict phsp with dropout (100 times)

# For each model type:
# Load model, and get test values
# Get mean value for each location
# Plot
lat_N=-45
lat_S=-90
lon_E=180
lon_W=-180
#%%
model_list = ['linear_model_phsp','linear_model_sili','dnn_model_phsp','dnn_model_sili']

for model_dir in ['dnn_model_phsp']:
#    model_dir = 'dnn_model_phsp'
    ml_model = ModelLoader(model_dir)
    predictions_lst = []
    #pred_array = np.zeros(100, data.shape[0])
    for i in range(100):
        test_predictions = ml_model.predict(testing_features).flatten()
        predictions_lst.append(test_predictions)
        #pred_array[i,:]=test_predictions 
        
    # get means and standard deviations
    arr_val = np.array(predictions_lst)
    mean_val = np.mean(arr_val, axis=0)
    sd_val = np.std(arr_val, axis=0)
    
    if model_dir == 'linear_model_phsp' or model_dir == 'dnn_model_phsp':
        true_data = data.loc[:,'phsp']
    elif model_dir == 'linear_model_sili' or model_dir == 'dnn_model_sili':
        true_data = data.loc[:,'sili']
    
    diff_vals = true_data - mean_val
 
#%%
    plt.rcParams.update({'font.size': 12})

    #plt.figure(figsize=(8,8))
    fig, axs = plt.subplots(1,3,figsize=(10,20))
    ax=plt.subplot(3,1,1)
    #ax.coastlines('50m')
    #ax.add_feature(ct.feature.LAND)
    #ax.add_feature(ct.feature.OCEAN)
    #ax.set_extent([lon_W, lon_E, lat_S, lat_N],ccrs.PlateCarree())
    cc= ax.scatter(lon, lat,c=true_data,s=1)
    #plt.gca().colorbar()
    fig.colorbar(cc)
    title_str = 'ESM Values (Phosphate)'
    plt.title(title_str)
    
    #fig, axs = plt.subplots(1,3,figsize=(8,8))
    ax=plt.subplot(3,1,2)
    #ax.coastlines('50m')
    #ax.add_feature(ct.feature.LAND)
    #ax.add_feature(ct.feature.OCEAN)
    #ax.set_extent([lon_W, lon_E, lat_S, lat_N],ccrs.PlateCarree())
    cc= ax.scatter(lon, lat,c=mean_val,s=1)
    #plt.gca().colorbar()
    fig.colorbar(cc)
    title_str = 'NN Predicted Values from ESM Features (Phosphate)'
    plt.title(title_str)
    
    #fig, axs = plt.subplots(1,3,figsize=(8,8))
    ax=plt.subplot(3,1,3)
    #ax.coastlines('50m')
    #ax.add_feature(ct.feature.LAND)
    #ax.add_feature(ct.feature.OCEAN)
    #ax.set_extent([lon_W, lon_E, lat_S, lat_N],ccrs.PlateCarree
    cc= ax.scatter(lon, lat,c=diff_vals,s=1,cmap='bwr')
    #plt.gca().colorbar()
    fig.colorbar(cc)
    title_str = 'Difference between ESM Oct-May Data (in training range) and NN Model Predictions (Phosphate)'
    plt.title(title_str)
    
#%%


for model_dir in ['dnn_model_sili']:
#    model_dir = 'dnn_model_phsp'
    ml_model = ModelLoader(model_dir)
    predictions_lst = []
    #pred_array = np.zeros(100, data.shape[0])
    for i in range(100):
        test_predictions = ml_model.predict(testing_features).flatten()
        predictions_lst.append(test_predictions)
        #pred_array[i,:]=test_predictions 
        
    # get means and standard deviations
    arr_val = np.array(predictions_lst)
    mean_val = np.mean(arr_val, axis=0)
    sd_val = np.std(arr_val, axis=0)
    
    if model_dir == 'linear_model_phsp' or model_dir == 'dnn_model_phsp':
        true_data = data.loc[:,'phsp']
    elif model_dir == 'linear_model_sili' or model_dir == 'dnn_model_sili':
        true_data = data.loc[:,'sili']
#%%
    diff_vals = true_data - mean_val

    plt.rcParams.update({'font.size': 12})


    #plt.figure(figsize=(8,8))
    fig, axs = plt.subplots(1,3,figsize=(10,20))
    ax=plt.subplot(3,1,1)
    #ax.coastlines('50m')
    #ax.add_feature(ct.feature.LAND)
    #ax.add_feature(ct.feature.OCEAN)
    #ax.set_extent([lon_W, lon_E, lat_S, lat_N],ccrs.PlateCarree())
    cc= ax.scatter(lon, lat,c=true_data,s=1,vmax=max(true_data))
    #plt.gca().colorbar()
    fig.colorbar(cc)
    title_str = 'ESM Values (Silicate)'
    plt.title(title_str)
    #plt.subplots_adjust(wspace=0.5,hspace=0.5)
    
    #fig, axs = plt.subplots(1,3,figsize=(8,8))
    ax=plt.subplot(3,1,2)
    #ax.coastlines('50m')
    #ax.add_feature(ct.feature.LAND)
    #ax.add_feature(ct.feature.OCEAN)
    #ax.set_extent([lon_W, lon_E, lat_S, lat_N],ccrs.PlateCarree())
    cc= ax.scatter(lon, lat,c=mean_val,s=1,vmax=max(true_data))
    #plt.gca().colorbar()
    fig.colorbar(cc)
    title_str = 'NN Predicted Values from ESM Features (Silicate)'
    plt.title(title_str)
    
    #fig, axs = plt.subplots(1,3,figsize=(8,8))
    ax=plt.subplot(3,1,3)
    #ax.coastlines('50m')
    #ax.add_feature(ct.feature.LAND)
    #ax.add_feature(ct.feature.OCEAN)
    #ax.set_extent([lon_W, lon_E, lat_S, lat_N],ccrs.PlateCarree
    cc= ax.scatter(lon, lat,c=diff_vals,s=1,cmap='bwr')
    #plt.gca().colorbar()
    fig.colorbar(cc)
    title_str = 'Difference between ESM Oct-May Data (in training range) and NN Model Predictions (Phosphate)'
    plt.title(title_str)

### not sure how we would plot the confidence intervals in this case
### there is no truth values

# # plot confidence intervals
# min_val = np.min(test_phsp)
# max_val = np.max(test_phsp)
# fig, ax = plt.subplots()
# diag_line = ax.plot(np.linspace(min_val,max_val,100),np.linspace(min_val,max_val,100), 'k--', alpha=0.2)
# ax.plot(np.sort(test_phsp), np.sort(linear_phsp_mean))
# ax.fill_between(np.sort(test_phsp), np.sort(linear_phsp_mean - 1.96*linear_phsp_sd), 
#                 np.sort(linear_phsp_mean + 1.96*linear_phsp_sd), alpha=0.2)
# plt.xlabel('True Values [phsp]')
# plt.ylabel('Predictions [phsp]')