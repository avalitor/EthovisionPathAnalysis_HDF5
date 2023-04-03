# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 01:30:02 2023

convert Kelly MAT mouse trial files to Mauricio MAT format

@author: Kelly
"""

import os
import numpy as np
import scipy.io
from modules.config import ROOT_DIR


mat_file1 = os.path.join(ROOT_DIR, 'data', 'processedData', '2022-10-11', 'hfm_2022-10-11_M4_Habituation 3.mat')
kelly_data = scipy.io.loadmat(mat_file1)
        

'''
bkgd_img --> arena_picture
eth_file --> file_trial_idx & trial_id
entrance --> start_location

arena_diameter = 120
unit_direction = deg
unit_r = cm
unit_time = s
unit_velocity = cm/s

MISSING INFORMATION:
direction
exper_date
file_name
is_reverse = 0
r_arena_center
r_arena_holes
r_start (coordinates)
r_target_alt
start_quadrant (1-4)
trial_name (Trial____37)

'''
kelly_data['arena_pic_bottom'] = kelly_data['img_extent'][0][2]
kelly_data['arena_pic_left'] = kelly_data['img_extent'][0][0]
kelly_data['arena_pic_right'] = kelly_data['img_extent'][0][1]
kelly_data['arena_pic_top'] = kelly_data['img_extent'][0][3]
del kelly_data['img_extent']

kelly_data['arena_picture'] = kelly_data['bkgd_img']
del kelly_data['bkgd_img']
kelly_data['file_trial_idx'] = kelly_data['eth_file']
del kelly_data['eth_file']
kelly_data['start_location'] = kelly_data['entrance']
del kelly_data['entrance']

kelly_data['arena_diameter'] = 120.
kelly_data['unit_direction'] = 'deg'
kelly_data['unit_r'] = 'cm'
kelly_data['unit_time'] = 's'
kelly_data['unit_velocity'] = 'cm/s'

# scipy.io.savemat(f"mpos_22Jun2021_trial_{kelly_data['eth_file']}_startloc_{kelly_data['start_location']}_day_{kelly_data['day']}.mat",kelly_data)
