# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:32:02 2023

get experiment information

@author: Kelly
"""
import numpy as np
import glob
import os
import pandas as pd
from modules import lib_process_data_to_mat as plib
import modules.config as cfg

def list_all_exp_by_date():
    
    pass

def get_exp_info(date):
    '''Get mouse number, trial schedule'''
    schedule = pd.DataFrame(columns=["Mouse","Day","Trial"])
 
    for files in os.listdir(glob.glob(glob.glob(cfg.PROCESSED_FILE_DIR+'/'+date+'/')[0], recursive = True)[0]): #finds file path based on experiment
        d = plib.TrialData()
        # print(files.split('_')[-2].split('.')[0][1:] + "  " + files.split('_')[-1].split('.')[0])
        d.Load(date, files.split('_')[-2].split('.')[0][1:], files.split('_')[-1].split('.')[0])
        
            
        # schedule.at[int(d.day), d.mouse_number] = d.trial
        # schedule.loc[int(d.day), d.mouse_number] = d.trial
        
        # schedule.loc[int(d.day)] = {"Trial": d.trial, "Mouse": d.mouse_number}
        
        # schedule = pd.concat([schedule.iloc[:d.day], line, schedule.iloc[d.day:]]).reset_index(drop=True)
        row = [int(d.mouse_number), int(d.day), d.trial]
        schedule.loc[len(schedule)] = row
        
        # schedule = schedule.sort_index().reset_index(drop=True)
        schedule = schedule.sort_values(by=['Day', 'Mouse']).reset_index(drop=True)
    return schedule

schedule = get_exp_info("2024-02-15")
