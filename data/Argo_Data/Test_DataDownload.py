#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 07:33:29 2021

@author: Ellen
"""

import GetArgoFiles as GAF

# GAF.GetArgoData(RegionOI=[55, 50, -15, -20], Ocean='A', FloatType='Both', SensorTypes =[],
#             ArgoDacDir='/Users/Ellen/Desktop/ArgoGDAC/', SaveFileDir='/Users/Ellen/Documents/Python/StartUp/GetArgoData_TextFiles/')

GAF.WriteBashFiles(TextDir='/Users/Ellen/Documents/Python/StartUp/GetArgoData_TextFiles/',
                   SaveFileDir='/Users/Ellen/Documents/Python/StartUp/GetArgoData_BashFiles/',
                   ArgoDacDir='/Users/Ellen/Documents/Python/StartUp/miniDac')
