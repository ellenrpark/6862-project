
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:10:59 2021
​
@author: Ellen
"""
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import matplotlib.pyplot as plt
from scipy import interpolate 

def DetermineAdjusted(raw_data, raw_data_QC,a_data, a_data_QC):
    # raw_data = not adjusted BGC Argo data
    # a_data = adjuasted BGC Argo data
    # Note! Update this function copying coreargo_fxns
    search_flag=0
    j=0
    adj_flag=np.NaN
    
    raw_ss=raw_data[0,:]
    a_ss=a_data[0,:]
    
    while (j < len(raw_ss) and search_flag == 0):
        
        if (np.isnan(raw_ss[j]) != True and np.isnan(a_ss[j]) != True):
            search_flag=1
            adj_flag=1
        elif (np.isnan(raw_ss[j]) != True and np.isnan(a_ss[j]) == True):
            search_flag=1
            adj_flag=0
        elif (np.isnan(raw_ss[j]) == True and np.isnan(a_ss[j]) == True):
            j=j+1
    
    if adj_flag == 1:
        output_data = a_data
        output_data_QC = a_data_QC
    elif adj_flag == 0:
        output_data = raw_data
        output_data_QC= raw_data_QC
    elif np.isnan(adj_flag) == True:
        print('ERROR: Using raw data')
        output_data=raw_data
        output_data_QC=raw_data_QC
        
    return output_data, output_data_QC, adj_flag



def ArgoQC(Data, Data_QC, goodQC_flags):
    
    AllQCLevels=[b'1',b'2',b'3',b'4',b'5',b'6',b'7',b'8',b'9']
    AllQCLevels_i=[1,2,3,4,5,6,7,8,9]
    QCData=np.zeros((Data.shape))
    
    for i in goodQC_flags:
        #print(i)
        AllQCLevels_i.remove(i)
    
    if len(Data.shape)==2:
        for i in np.arange(Data.shape[0]):
            
            t_df=pd.DataFrame({'Data':Data[i,:],'QC':Data_QC[i,:]})
            #print(t_df)
            for j in AllQCLevels_i:
                qc_l=AllQCLevels[j-1]
                t_df.loc[t_df.loc[:,'QC']==qc_l,'Data']=np.NaN
            
            t_df = t_df.loc[:,'Data'].to_numpy()
            QCData[i,:]=t_df
    
    return QCData


#adjusted for file location on Nadège's computer
def ArgoDataLoader(WMO, DAC):
    BGCfile='/Volumes/D13_1/Argo_Data/dac/'+DAC+'/'+str(WMO)+'/'+str(WMO)+'_Sprof.nc'
    return xr.open_dataset(BGCfile)

    
def WMODacPair():
    PairFile = '/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/DacWMO_NAtlantic.txt'
    dacs = []
    wmos = []
    
    count = 0
    with open(PairFile) as fp:
        Lines = fp.readlines()
        for line in Lines:
            count += 1
            x=line.strip()
            xs = x.split('/')
            dacs=dacs+[xs[0]]
            wmos = wmos + [xs[1]]
    
    Dict = {}
    for i in np.arange(len(dacs)):
        Dict[int(wmos[i])]=dacs[i]
    
    return Dict