#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 07:33:29 2021

@author: Ellen
"""

import DP_GetArgoFilesFxns as GAF

# Where is your local GitHub repo?
GitHubDir='/Users/Ellen/Documents/GitHub/6862-project/'

# Where on your local computer (NOT IN THE GITHUB REPO) do you want to download your data?
LocalArgoDacDir='/Users/Ellen/Desktop/ArgoGDAC/'

# Ocean region 
LatN=-45
LatS=-90
LonE=180
LonW=-180

GAF.GetArgoData(RegionOI=[LatN, LatS, LonE, LonW], FloatType='Both', SensorTypes =[],
            ArgoDacDir=LocalArgoDacDir, SaveFileDir=GitHubDir+'data/Argo_Data/GetArgoData_TextFiles/')

GAF.WriteBashFiles(GitHubDir+'data/Argo_Data/GetArgoData_TextFiles/',
                   SaveFileDir=GitHubDir+'data/Argo_Data/GetArgoData_BashFiles/',
                   ArgoDacDir=LocalArgoDacDir)
