# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:18:36 2022

automatically processes ethovision files from one experiment into mat files
requires the following modules:
    lib_process_data_to_mat

Parameter files that need to be updated per experiment:
    modules.lib_experiment_parameters
    data/processedData/documentation.csv

@author: Kelly
"""
import os
import modules.lib_process_data_to_mat as plib
from modules.config import RAW_FILE_DIR

experiment = '2019-12-11'
i=1 #starts at this ethovision file

#iterates over all files in experiment folder and saves as mat
for f in os.listdir(os.path.join(RAW_FILE_DIR, experiment+'_Raw Trial Data')):
    
    # data = plib.get_excel_data(experiment, i)
    # data.Store()
    print(i , f)
    i = i+1

    