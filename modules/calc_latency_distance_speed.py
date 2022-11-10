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
import scipy.stats as stats
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

#gets trajectory coordinates between two points
def get_coords_bw_points(coords, pointA, pointB):
    idx_start = pltlib.coords_to_target(coords, pointA)+1
    idx_end = pltlib.coords_to_target(coords, pointB)+1
    if idx_start < idx_end: #checks if first point comes before second point
        coord_interval = coords[idx_start:idx_end]
    else:
        coord_interval = coords[idx_end:idx_start]
    return coord_interval

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


def curve_pValue(data):
    
    def early_late_ttest(stat_data):
        early = [x for xs in stat_data.iloc[1:4,:].values.tolist() for x in xs] #get trials 2-4
        late = [x for xs in stat_data.tail(3).values.tolist() for x in xs] #get last 3 trials
        t_test = stats.ttest_rel(early, late, nan_policy = 'omit') #Calculate the t-test on TWO RELATED samples of scores
        return t_test
    
    pValues = []
    for l in data:
        p = early_late_ttest(data[l])[1]
        pValues.append(p)
        if p < 0.05:
            print('%s is significant at p = %s'%(l, round(p, 6)))
        else:
            print('%s is NOT significant at p = %s'%(l, round(p, 6)))
    # pValues = early_late_ttest(latency)[1], early_late_ttest(distance)[1], early_late_ttest(speed)[1]
    
    return pValues


#%%
'''
Calculate Search bias during probe
'''

def calc_time_in_area(target, array, idx_end, radius=15.):
    '''
    Calculates amount of time spent within a certain radius of the target (area is actually a square)
    
    Parameters
    ----------
    target : tuple
        x y coordinate at the center of the region of interest.
    array : 2xn array
        list of xy coordinates covering the trajectory of interest.
    idx_end : int
        index of array where you want to cut it off, usually a time limit of 2min
    radius : float, optional
        Radius from the target coordinate that you want to include. The default is 15cm.

    Returns
    -------
    time_in_range : float
        time spent within a certain range of the target coordinate, in seconds

    '''
    x = target[0]
    y = target[1]
    array = np.array(array[:idx_end], dtype=np.float64)
    #CHECK OUT lib_plot_mouse_trajectory for coords_to_target function to get a circle
    coords_in_range = ((array >= [x-radius, y-radius]) & (array <= [x+radius, y+radius])).all(axis=1) #gets list of coordinates within range
    idx_in_range = np.where(abs(coords_in_range)==True)[0] #gets index of the coordinates in range
    
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

'''
Calculat spread along symmetry line betwen two points
'''


def axis_of_symm(A, B): #gets axis of symmetry between two points
    midpoint = ((A[0] + B[0])/2, (A[1] + B[1])/2)
    theta = np.polyfit(np.array([A[0], B[0]]), np.array([A[1], B[1]]), 1) #gets line between two points
    slope = (-1/theta[0]) #gets the perpendicular slope
    b = midpoint[1] - (-1/theta[0]) * midpoint[0] #calculates perpendicular line bewteen points
    return slope,  b

#line = a list of slope and b
def get_perp_intersect(point, line): #gets coords where point perpendicularly intersects with line
    point_m = -1/line[0]
    point_b = point[1]-(point_m*point[0])
    
    x = (point_b - line[1])/(line[0]-point_m)
    y = line[0]*x + line[1]
    return x, y


def dist_from_optimal_path(point, targetA, targetB):
    optimal_path = np.polyfit(np.array([targetA[0], targetB[0]]), np.array([targetA[1], targetB[1]]), 1) #gets line between two points
    
    line_intersect = get_perp_intersect(point, optimal_path)
    
    distance = pltlib.calc_dist_bw_points(point, line_intersect)
    return distance

def calc_traj_spread(exp):
    targetA = exp.target
    targetB = exp.target_reverse
    
    coord_interval = get_coords_bw_points(exp.r_nose, targetB, targetA)
    
    i = 0
    point_distances = np.zeros([len(coord_interval), 1])
    for c in coord_interval:
        point_distances[i] = dist_from_optimal_path(c, targetA, targetB) #NEED TO FIX THIS
        i = i+1
        
    max_distance = np.max(point_distances)
    
    # def myfunc(n):
    #     for i in n:
    #         yield i #get_perp_intersect(i, symm_line)
            
    # moved_points = np.fromiter(myfunc(coord_interval), dtype=float)
    
    # i = 0
    # moved_points = np.empty([len(coord_interval), 2])
    # while i < len(coord_interval):
    #     moved_point = get_perp_intersect(coord_interval[i], symm_line)
    #     moved_points[i] = moved_point
    #     i = i+1
        
    return max_distance

def calc_spread_iterate_exp(experiments, show_load = True):
    pass

if __name__ == '__main__':
    exp = plib.TrialData()
    exp.Load('2022-10-11', '3', 'Probe 2')
    
    coord_interval = get_coords_bw_points(exp.r_nose, exp.target_reverse, exp.target)
    
    # test2 = dist_from_optimal_path((0,0), exp.target, exp.target_reverse)
    
    max_distance = calc_traj_spread(exp)
    
    
