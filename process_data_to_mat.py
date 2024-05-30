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
import modules.lib_plot_mouse_trajectory as pltlib
import modules.cv_arena_hole_detect as phc
import modules.calc_latency_distance_speed as calc
from plot_hole_checks import main_wrap_get_time
import modules.config as cfg

#%%
'''Whole experiment import'''

experiment = '2024-05-06'
i=1 #starts at this ethovision file

#iterates over all files in experiment folder and saves as mat
for f in os.listdir(os.path.join(cfg.RAW_FILE_DIR, experiment+'_Raw Trial Data')):
    
    data = plib.get_excel_data(experiment, int(f.split('  ')[-1].split('.')[0])) #gets the ethovision file number
    data.Store()
    print(i , f)
    i = i+1
    
    
#%% Run this cell as a check first
'''Hole Detection'''

# experiment = '2023-10-16'

exp = plib.TrialData()
exp.Load(experiment, '*', '17') #use this trial's image as the standard
arena_circle, gray = phc.detect_arena_circle(os.path.join(cfg.ROOT_DIR, 'data', 'BackgroundImage', exp.bkgd_img), 
                                             mask_sensitivity=60.)
holes = phc.detect_arena_hole_coords(arena_circle, gray)
r_holes, arena_coords = phc.transpose_coords(holes, arena_circle, gray.shape, exp.img_extent)

# holes_filt = phc.filter_holes(holes, arena_circle, gray, threshold=(10,10)) #optional filtering

# exp.Load(experiment, '73', 'Probe2')
# exp.arena_circle = arena_coords
# exp.r_arena_holes = r_holes
# exp.Update()


#%% run this cell to update all files with circle and hole coordinate data (make sure to check the detection is correct first)
for files in os.listdir(os.path.join(cfg.PROCESSED_FILE_DIR, experiment)):
    # print(files)
    d = plib.TrialData()
    d.Load(experiment, files.split('_')[-2].split('.')[0][1:], files.split('_')[-1].split('.')[0])
   
    d.arena_circle = arena_coords
    d.r_arena_holes = r_holes
    d.Update()

#%% update files with hole check times and reward get time
# experiment = '2023-12-18'
# d = plib.TrialData()
# d.Load(experiment, '101', 7)
# holes = main_wrap_get_time(d)

for files in os.listdir(os.path.join(cfg.PROCESSED_FILE_DIR, experiment)):
    # print(files)
    d = plib.TrialData()
    d.Load(experiment, files.split('_')[-2].split('.')[0][1:], files.split('_')[-1].split('.')[0])
   
    d.k_hole_checks = main_wrap_get_time(d)[1]
    d.k_reward = pltlib.coords_to_target(d.r_nose, d.target)
    d.heading = calc.heading(d)
    print (f'Mouse {d.mouse_number} Trial {d.trial} Reward Idx: {d.k_reward}')
    d.Update()
#%%
'''Single Trial Import'''

trialdict={ #need to manually input values
  "mouse_number": "Nas1",
  "mouse_sex": "male",
  "trial": '1',
  "protocol_name": 'LED_Test',
  'protocol_description': "Nastaran's implanted mouse explores an arena with cheerios for 30min",
  'img_extent': '-137.24, 135.96, -76.82, 76.82',
  'bkgd_img': 'BKGD_2022-09-12_LED Test merged.png'
}

datum = plib.manual_single_excel_import(
    (cfg.RAW_FILE_DIR+'\Single Trial Data\Raw data-LED_test-Trial     3.xlsx'),trialdict)
datum.Update()

    