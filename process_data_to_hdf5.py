# -*- coding: utf-8 -*-
"""
Created on Thu May  5 18:12:29 2022

library to process excel data int hdf5 files

@author: Kelly
"""

import os
import h5py
import numpy as np
import scipy.io
import pandas as pd
import glob #for file search

from modules.config import ROOT_DIR

# raw_data_path = os.path.join(ROOT_DIR, 'data', 'processedData', 'ms6240_2022-03-01_block1.hdf5')

# savepath = os.path.join(ROOT_DIR, 'data', 'processedData', 'test.hdf5')


mat_file2 = os.path.join(ROOT_DIR, 'data', 'processedData', '2021-06-22', 'hfm_2021-06-22_M34_Probe2.mat')
# excel_file = os.path.join(ROOT_DIR, 'data', 'rawData', '2021-07-16_Raw Trial Data', 'Raw data-Hidden Food Maze-16Jul2021-Trial    25.xlsx')

# exp = '2021-07-16'
# eth_file = 25


def import_hdf5(filepath):
    with h5py.File(filepath, 'r') as f: #use the with function to automatically close the file after getting the attributes
        # Get relevant information from the data file.
        prev_protocol = f.attrs['protocol_name']
        prev_user = f.attrs['experimenter']
    return f

# data_hdf5 = import_hdf5(raw_data_path)
# data_np = np.array(data_hdf5.get("t_end"), dtype=np.float64)

        
def get_filename(path): #gets file name from full path
    return os.path.split(path)[1]

def get_file_from_exp(exp, eth_file):
    path = glob.glob(glob.glob(ROOT_DIR+'/data/rawData/'+exp+'*/')[0]+'*'+str(eth_file)+'.xlsx', recursive = True)[0] #finds file path based on ethovision trial number
    return os.path.split(path)[1]



def store_as_hdf5(filepath):
    f = h5py.File(filepath,'w') #creates file 
    grp_meta = f.create_group("meta") #creates group
    grp_coords = f.create_group("data/coordinates") #creates nested groups
    coords_nose = grp_coords.create_dataset("nose", (10,2), 'f')
    return

# store_as_hdf5(savepath)


#loads mat file

mat_data2 = scipy.io.loadmat(mat_file2)