#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:20:23 2021

@author: Ellen
"""
import cartopy as ct
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import numpy as np

GitHubDir='/Users/Ellen/Documents/GitHub/6862-project/'
GOSHIPData=pd.read_csv(GitHubDir+'data/GOSHIP_Data/QCFilteredData.csv')
FigDir=GitHubDir+'figures/GOSHIPDataOverview/'

lat_N=-45
lat_S=-90
lon_E=180
lon_W=-180

lat=GOSHIPData.loc[:,'LATITUDE']
lon=GOSHIPData.loc[:,'LONGITUDE']

plt.figure(figsize=(8,8))
ax=plt.subplot(1,1,1,projection=ccrs.SouthPolarStereo())
#ax.coastlines('50m')
ax.add_feature(ct.feature.LAND)
ax.add_feature(ct.feature.OCEAN)
ax.set_extent([lon_W, lon_E, lat_S, lat_N],ccrs.PlateCarree())
ax.scatter(lon, lat,c='red',s=1,transform=ccrs.PlateCarree())
plt.title('GO-SHIP Transects')
plt.savefig(FigDir+'GOSHIPData_map.jpg')
plt.clf(); plt.close()

GOSHIPData.hist('MONTH',figsize=(8,8))
plt.ylabel('# Data points')
plt.xlabel('Month')
plt.title('GO-SHIP All Data Points Histogram (n = '+str(GOSHIPData.shape[0])+')')
plt.savefig(FigDir+'GOSHIPData_histogram_alldata.jpg')
plt.clf(); plt.close()

## Plot only surface values
pres_ranges=[10,100,500, 1000]
Months=['Jan','Feb','March','April','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

var_list=['TEMP','SAL','OXY','NITR','PHSP','SILI']
var_list_full=['Temperature', 'Salinity','Oxygen','Nitrogen','Phosphate','Silicate']
units=['ºC','PSU','µmol/kg','µmol/kg','µmol/kg','µmol/kg']
    
for p in np.arange(len(pres_ranges)):
    min_surf_P=pres_ranges[p]
    
    GOSHIPData=pd.read_csv(GitHubDir+'data/GOSHIP_Data/QCFilteredData.csv')
    GOSHIPData.loc[GOSHIPData['PRES']>=min_surf_P,:]=np.NaN
    GOSHIPData=GOSHIPData.dropna()
    
    GOSHIPData.hist('MONTH',figsize=(8,8))
    plt.ylabel('# Data points')
    plt.xlabel('Month')
    plt.title('GO-SHIP Data (P < '+str(min_surf_P)+' dbar) Histogram (n = '+str(GOSHIPData.shape[0])+')')
    plt.savefig(FigDir+'GOSHIPData_histogram_maxP'+str(min_surf_P)+'dbar.jpg')
    plt.clf(); plt.close()
    
    lat_surf=GOSHIPData.loc[:,'LATITUDE']
    lon_surf=GOSHIPData.loc[:,'LONGITUDE']
    
    ## Attach nan filler data so all months are present
        
    MeanSurfVals=GOSHIPData.groupby('MONTH').mean()
    StdSurfVals=GOSHIPData.groupby('MONTH').std()
         
    for i in np.arange(len(var_list)):
        
        # Spatial data
        var_name=var_list[i]
        plt.figure(figsize=(8,8))
        ax=plt.subplot(1,1,1,projection=ccrs.SouthPolarStereo())
        #ax.coastlines('50m')
        ax.add_feature(ct.feature.LAND)
        ax.add_feature(ct.feature.OCEAN)
        ax.set_extent([lon_W, lon_E, lat_S, lat_N],ccrs.PlateCarree())
        A=ax.scatter(lon_surf, lat_surf,c=GOSHIPData.loc[:,var_name],s=1,transform=ccrs.PlateCarree())
        plt.colorbar(A)
        plt.title('GO-SHIP Values (P < '+str(min_surf_P)+' dbar) Map: '+var_list_full[i]+' ('+units[i]+')')
        plt.savefig(FigDir+'GOSHIPData_map_maxP'+str(min_surf_P)+'dbar_'+var_list_full[i]+'.jpg')
        plt.clf(); plt.close()
        
        # Surface Time Series
        m_list=MeanSurfVals.index.to_list()
        mean_vals=MeanSurfVals.loc[:,var_name].to_numpy()
        std_vals=StdSurfVals.loc[:,var_name].to_numpy()
        
        mean_reform=np.zeros(12)
        mean_reform[:]=np.NaN
        std_reform=np.zeros(12)
        std_reform[:]=np.NaN
        
        for j in np.arange(len(m_list)):
            m_ind=int(m_list[j]-1)
            mean_reform[m_ind]=mean_vals[j]
            std_reform[m_ind]=std_vals[j]
            
        plt.figure(figsize=(8,8))
        plt.errorbar(Months, mean_reform,yerr=std_reform)
        plt.title('GO-SHIP Monthly Averages (P < '+str(min_surf_P)+' dbar): '+var_list_full[i])
        plt.ylabel(var_list_full[i]+' ('+units[i]+')')
        plt.xticks(rotation=45)
        plt.savefig(FigDir+'GOSHIPData_month_maxP'+str(min_surf_P)+'dbar_'+var_list_full[i]+'.jpg')
        plt.clf(); plt.close()

plt.show()
