# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:17:26 2022

Analyzes trial data

@author: Kelly
"""
import numpy as np
import glob
import os
import pandas as pd
from modules import lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib
from modules.config import PROCESSED_FILE_DIR

'''
Single Trial Calculation Functions
'''
#calculates path length to target, coordinates need to be pre-cropped
def calc_distance(coords):
    x, y = coords[:,0], coords[:,1]
    dist = [np.sqrt((x[n]-x[n-1])**2 + (y[n]-y[n-1])**2) for n in range(1,len(x))]
    distance = np.nansum(dist)
    return distance

#calculates speed, coordinates need to be pre-cropped
def calc_speed(coords, time):
    speed = calc_distance(coords)/np.nansum(time)
    return speed

def calc_lat_dist_sped(data, continuous = False):
    # data = plib.TrialData()
    # data.Load(exp, mouse, trial)
    coords = data.r_nose #choose which body coordinate you want, change to r_nose or r_center for different coordinates
    
    idx_end = pltlib.coords_to_target(coords, data.target)+1 #plus 1 so it includes the indexed coordinate
    
    if continuous == True: 
        idx_start = pltlib.continuous_coords_to_target(data, idx_end)
    else: idx_start = 0
    
    latency = data.time[0,idx_end] - data.time[0,idx_start] #calculates total latency before reaching target
    distance = calc_distance(coords[idx_start:idx_end]) #remember nose point distance is longer than center point distance
    speed = calc_speed(coords[idx_start:idx_end], latency)
    return latency, distance, speed

# lat, dist, speed = calc_lat_dist_sped('2021-07-16', 38, 3, continuous = False)

#calculate trajectory distance between two points
def calc_dist_bw_points(coords, pointA, pointB):
    idx_start = pltlib.coords_to_target(coords, pointA)+1
    idx_end = pltlib.coords_to_target(coords, pointB)+1
    if idx_start < idx_end: #checks if first point comes before second point
        dist = calc_distance(coords[idx_start:idx_end])
    else:
        dist = calc_distance(coords[idx_end:idx_start])
    return dist
    

#%%
'''
Iterate over whole experiment
'''

def get_probe_day(experiment):
    d = plib.TrialData()
    d.Load(experiment, '*', 'Probe')
    probe_day = d.day
    return probe_day

def iterate_all_trials(experiment, continuous = False, show_load = True):
    latency = pd.DataFrame()
    distance = pd.DataFrame()
    speed = pd.DataFrame()
    
    if type(experiment) is not list: experiment = [ experiment ] #checks if it is a list of experiments
    for exp in experiment:
        for files in os.listdir(glob.glob(glob.glob(PROCESSED_FILE_DIR+'/'+exp+'/')[0], recursive = True)[0]): #finds file path based on experiment
            if files.split('_')[-1].split('.')[0].isdigit(): #checks if it is a training file
                d = plib.TrialData()
                d.Load(exp, files.split('_')[-2].split('.')[0][1:], files.split('_')[-1].split('.')[0])
    
                if show_load: print('Reading M%s Trial %s'%(d.mouse_number, d.trial))
                if int(d.day) < int(get_probe_day(exp)): #checks if training trial took place before probe day
    
                    latency.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[0]
                    distance.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[1]
                    speed.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[2]
    
    return {'Latency': latency.sort_index(axis=0), 'Distance': distance.sort_index(axis=0),'Speed': speed.sort_index(axis=0)}  #sorts trials into the correct order and saves as combined dict

#%%
'''
Calculate Search bias during probe
'''
#Calculates amount of time spent within a certain radius of the target (area is actually a square)
def calc_time_in_area(target, array, idx_end, radius=15.):
    x = target[0]
    y = target[1]
    array = np.array(array[:idx_end], dtype=np.float64)
   
    coords_in_range = ((array >= [x-radius, y-radius]) & (array <= [x+radius, y+radius])).all(axis=1)
    idx_in_range = np.where(abs(coords_in_range)==True)[0]
    
    time_in_range = len(idx_in_range)*0.0333333 #multiplies #samples with sampling rate (30fps) to get time
    return time_in_range

#compares target dwell with REL targets, this only works with probe trials, so the center can be found
def compare_target_dwell(data, time_limit = '2min'):
    idx_end = pltlib.get_coords_timeLimit(data, time_limit)
    t1 = calc_time_in_area(data.target, data.r_nose, idx_end, 15.)
    t2 = calc_time_in_area(pltlib.rotate(data.target, pltlib.get_arena_center(data.r_nose), 270), data.r_nose, idx_end, 15.)
    t3 = calc_time_in_area(pltlib.rotate(data.target, pltlib.get_arena_center(data.r_nose), 180), data.r_nose, idx_end, 15.)
    t4 = calc_time_in_area(pltlib.rotate(data.target, pltlib.get_arena_center(data.r_nose), 90), data.r_nose, idx_end, 15.)
    return np.array([t1, t2, t3, t4])

#sums the dwell time of all probe trials in an experiment
def calc_search_bias(experiment, time_limit = '2min'):
    total_dwell = np.zeros(4)
    if type(experiment) is not list: experiment = [ experiment ] #checks if it is a list of experiments
    for exp in experiment:
        for files in os.listdir(glob.glob(glob.glob(PROCESSED_FILE_DIR+'/'+exp+'/')[0], recursive = True)[0]): #finds file path based on experiment
            if files.split('_')[-1].split('.')[0] == 'Probe': #checks if it is a probe file
                d = plib.TrialData()
                d.Load(exp, files.split('_')[-2].split('.')[0][1:], files.split('_')[-1].split('.')[0])
                total_dwell = total_dwell + compare_target_dwell(d, time_limit)
            
    return total_dwell


if __name__ == '__main__':
    # exp_data = iterate_all_trials('2021-07-16')
    d = plib.TrialData()
    d.Load('2021-07-16', '*', 'Probe')
    
    # test = compare_target_dwell(d, '2min')
    
    test2 = calc_search_bias(d.exp, '2min')
    
    # t1, coords = calc_time_in_area(d.target, d.r_nose, 3000, 15.)
    # d.r_nose = d.r_nose[idx]
    # pltlib.plot_single_traj(d, cropcoords=False)
