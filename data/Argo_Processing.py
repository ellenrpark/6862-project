#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:58:30 2021

@author: nsaoki
"""

import numpy as np
import pandas as pd
import xarray as xr
#import gsw
import matplotlib.pyplot as plt
from scipy import interpolate
import RandomFxns as RF
import time
from datetime import datetime

#BGCfile='/Volumes/D13_1/Argo_Data/dac/csiro/1901135/1901135_Sprof.nc'
#Data = xr.open_dataset(BGCfile)

#print(list(Data.keys()))

#%%

Argofiles='/Users/nsaoki/Desktop/MIT:WHOI/Classes/Spring 2021/6.862_6.036_ Machine_Learning/6862-project/data/argo_code_na/ArgoFiles.csv'

#SectionFigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/Sections/'

ArgoDataFrames=pd.read_csv(Argofiles)
ArgoDataFrames=ArgoDataFrames.dropna()
new_ind=np.arange(ArgoDataFrames.shape[0])
ArgoDataFrames=ArgoDataFrames.set_index(pd.Index(list(new_ind)))
    
ArgoWMO=ArgoDataFrames.loc[:,'FloatWMO']
FileNames=ArgoDataFrames.loc[:,'FileName']

#%%
good_QC=[1,2,5]

lat_N= -45
lat_S= -90
lon_E= 180
lon_W= -180

#%%
#def ArgoDataLoader(WMO, DAC):
#    BGCfile='/Volumes/D13_1/Argo_Data/dac/'+DAC+'/'+str(WMO)+'/'+str(WMO)+'_Sprof.nc'
#    Data = xr.open_dataset(BGCfile)


tic1=time.time()
bad_floats=[]
#d2={}


for i in np.arange(len(ArgoWMO)): 
    #finished 23 floats
    tic2=time.time()
    
    
    dac=FileNames[i].split('/')[0]
    WMO=int(ArgoWMO[i])
    
    d[WMO] = pd.DataFrame(columns = ['FLOAT','DATE','LATITUDE','LONGITUDE','PRES','TEMP','SAL','OXY','NIT','MONTH'])
    
    print('\n%%% ',WMO,' %%%\n')
    print(i, ' Floats completed; ', len(ArgoWMO)-i,' Floats Left')
    f = RF.ArgoDataLoader(DAC=dac, WMO=WMO)
    
    # Make sure float measures oxygen and nitrogen
    float_vars=list(f.keys())
    float_vars=np.array(float_vars)
    oxy_check = np.where(float_vars=='DOXY')
    nit_check = np.where(float_vars=='NITRATE')
    
    

       
    if oxy_check[0].size !=0 and nit_check[0].size != 0:
        print('Measured O and N')
        # Load float data and determine if use adjusted or not adjusted data
        [pres, pres_QC, pres_flag]=RF.DetermineAdjusted(raw_data=f.PRES.values,raw_data_QC=f.PRES_QC.values,a_data=f.PRES_ADJUSTED.values, a_data_QC=f.PRES_ADJUSTED_QC.values)
        [temp, temp_QC, temp_flag]=RF.DetermineAdjusted(raw_data=f.TEMP.values, raw_data_QC= f.TEMP_QC.values,a_data=f.TEMP_ADJUSTED.values, a_data_QC=f.TEMP_ADJUSTED_QC.values)
        [sal, sal_QC, sal_flag]=RF.DetermineAdjusted(raw_data=f.PSAL.values, raw_data_QC=f.PSAL_QC.values,a_data=f.PSAL_ADJUSTED.values,a_data_QC=f.PSAL_ADJUSTED_QC.values)
        [doxy, doxy_QC, doxy_flag]=RF.DetermineAdjusted(raw_data=f.DOXY.values, raw_data_QC=f.DOXY_QC.values,a_data=f.DOXY_ADJUSTED.values,a_data_QC=f.DOXY_ADJUSTED_QC.values)
        [nit, nit_QC, nit_flag]=RF.DetermineAdjusted(raw_data=f.NITRATE.values, raw_data_QC=f.NITRATE_QC.values,a_data=f.NITRATE_ADJUSTED.values,a_data_QC=f.NITRATE_ADJUSTED_QC.values)

        ## Quality control float data ##
        pres=RF.ArgoQC(Data=pres, Data_QC=pres_QC, goodQC_flags=good_QC)
        temp=RF.ArgoQC(Data=temp, Data_QC=temp_QC, goodQC_flags=good_QC)
        sal=RF.ArgoQC(Data=sal, Data_QC=sal_QC, goodQC_flags=good_QC)
        doxy=RF.ArgoQC(Data=doxy, Data_QC=doxy_QC, goodQC_flags=good_QC)
        nit=RF.ArgoQC(Data=nit, Data_QC=nit_QC, goodQC_flags=good_QC)

        lat=f.LATITUDE.values
        lon=f.LONGITUDE.values
        
        dates=f.JULD.values
        
        date_reform=[[]]*dates.shape[0]
        bad_index=[]
        
        for k in np.arange(len(date_reform)):
            if np.isnat(dates[k]) == False:
                date_reform[k]=datetime.fromisoformat(str(dates[k])[:-3])
            else:
                date_reform[k]=dates[k]
                bad_index=bad_index+[k]
        
        #check if location is in Southern Ocean
         # For each profile, determine if data point is in BC, Gyre, or N/A
        for j in np.arange(len(lat)):            
            print("Exporting Profile: "+str(j+1))
            # Make sure data point is not a bad index 
            bad_check = 0
            if len(bad_index)>0:
                for b_i in bad_index:
                    if b_i == j:
                        bad_check =1
             
            if bad_check == 0:          
                # Make sure profile is in the generally correct region and a valid position
                if (lat[j]<=lat_N and lat[j]>=lat_S and lon[j]>=lon_W and lon[j]<=lon_E and np.isnan(lat[j]) == False and np.isnan(lon[j]) == False):
                    # Save profile information 
                    #print("in SO")
                    a=1
                    
                    #print(nit[j,:])
                else:
                    #print("not in SO")
                    # Replace values with nan
                    lat[j]=np.NaN
                    lon[j]=np.NaN
                    pres[j,:]=np.NaN
                    temp[j,:]=np.NaN
                    sal[j,:]=np.NaN
                    doxy[j,:]=np.NaN
                    nit[j,:]=np.NaN
                    #print(nit[j,:])
            
            #d[WMO] = pd.DataFrame(columns = ['FLOAT','DATE','LATITUTE','LONGITUDE','PRES','TEMP','SAL','OXY','NITR','MONTH'])
            for z in range(len(pres[0])):
                #print("Depth: "+str(z+1))
                if np.isnan(lat[j]) or np.isnan(lon[j]) or np.isnan(pres[j,z]) or np.isnan(temp[j,z]) or np.isnan(sal[j,z]) or np.isnan(doxy[j,z]) or np.isnan(nit[j,z]):                    
                    pass
                else:
                    month = np.datetime_as_string(dates[j]).split('-')[1]
                    d[WMO] = d[WMO].append({'DATE':dates[j],'FLOAT':WMO,'LATITUDE':lat[j],'LONGITUDE':lon[j],
                                            'PRES':pres[j,z],'TEMP':temp[j,z],'SAL':sal[j,z],'OXY':doxy[j,z],'NIT':nit[j,z],'MONTH':int(month)},ignore_index=True)
      
tic3 = time.time()

print(tic3-tic1)

#%%

new_d ={}

for key in d:
    new_d[key] = d[key].dropna()

for data in new_d:
    new_d[data].to_csv(str(data)+'.csv')
