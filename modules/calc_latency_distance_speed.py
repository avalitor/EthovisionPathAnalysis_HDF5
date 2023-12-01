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

def calc_lat_dist_sped(data, custom_target = False, continuous = False):
    '''
    calculate latency distanca and speed of single trial

    Parameters
    ----------
    data : TrialData custom class
        Custom data class.
    custom_target : bool or tuple, optional
        Crop trajectory at custom coordinate that is not the target. The default is False.
    continuous : bool, optional
        Only calculate from the last continuous trajectory. The default is False.

    Returns
    -------
    latency : float64
        time to target or custom coordinate in seconds.
    distance : float64
        distance to target coordinate in cm.
    speed : float64
        average speed to custome coordinate in cm/s.

    '''
    # data = plib.TrialData()
    # data.Load(exp, mouse, trial)
    coords = data.r_nose #choose which body coordinate you want, change to r_nose or r_center for different coordinates
    
    if isinstance(custom_target, bool):
        idx_end = pltlib.coords_to_target(coords, data.target)+1 #plus 1 so it includes the indexed coordinate
    else:
        idx_end = pltlib.coords_to_target(coords, custom_target)+1 #plus 1 so it includes the indexed coordinate
    
    if continuous == True: 
        idx_start = pltlib.continuous_coords_to_target(data, idx_end)
    else: idx_start = 0
    
    latency = data.time[idx_end] - data.time[idx_start] #calculates total latency before reaching target
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
Iterate over whole experiment & significance test
'''

def get_probe_day(experiment):
    d = plib.TrialData()
    d.Load(experiment, '*', 'Probe')
    probe_day = d.day
    return probe_day

def iterate_all_trials(experiment, training_trials_only = True, continuous = False, show_load = True):
    latency = pd.DataFrame()
    distance = pd.DataFrame()
    speed = pd.DataFrame()
    
    if type(experiment) is not list: experiment = [ experiment ] #checks if it is a list of experiments
    for exp in experiment:
        for files in os.listdir(glob.glob(glob.glob(PROCESSED_FILE_DIR+'/'+exp+'/')[0], recursive = True)[0]): #finds file path based on experiment
            if training_trials_only:
                if files.split('_')[-1].split('.')[0].isdigit(): #checks if it is a training file
                    d = plib.TrialData()
                    d.Load(exp, files.split('_')[-2].split('.')[0][1:], files.split('_')[-1].split('.')[0])
        
                    if show_load: print('Reading M%s Trial %s'%(d.mouse_number, d.trial))
                    if int(d.day) < int(get_probe_day(exp)): #checks if training trial took place before probe day
        
                        latency.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[0]
                        distance.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[1]
                        speed.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[2]
            else: #NOT FINISHED, need to figure out how to exclude habituation trials
                if files.split('_')[-1].split('.')[0].isdigit(): #checks if it is a training file
                    d = plib.TrialData()
                    d.Load(exp, files.split('_')[-2].split('.')[0][1:], files.split('_')[-1].split('.')[0])
                
                if show_load: print('Reading M%s Trial %s'%(d.mouse_number, d.trial))
    
                latency.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[0]
                distance.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[1]
                speed.at[int(d.trial), d.mouse_number] = calc_lat_dist_sped(d, continuous)[2]
    
    return {'Latency': latency.sort_index(axis=0), 'Distance': distance.sort_index(axis=0),'Speed': speed.sort_index(axis=0)}  #sorts trials into the correct order and saves as combined dict


def curve_pValue(data):
    
    def early_late_ttest(stat_data):
        early = [x for xs in stat_data.iloc[0:3,:].values.tolist() for x in xs] #get trials 1-3
        late = [x for xs in stat_data.tail(3).values.tolist() for x in xs] #get last 3 trials
        t_test = stats.ttest_rel(early, late, nan_policy = 'omit') #Calculate the t-test on TWO RELATED samples of scores
        return t_test
    
    pValues = []
    for l in data: #does t-test for every key in dict (distace, latency, speed)
        p = early_late_ttest(data[l])[1]
        pValues.append(p)
        if p < 0.05:
            print('%s is significant at p = %s'%(l, round(p, 6)))
        else:
            print('%s is NOT significant at p = %s'%(l, round(p, 6)))
    # pValues = early_late_ttest(latency)[1], early_late_ttest(distance)[1], early_late_ttest(speed)[1]
    
    return pValues


def compare_curve_pValue(data1, data2): #input must be dataframes of the same size
    t_test = stats.ttest_ind(data1, data2, axis=1, nan_policy = 'omit') #Calculate the t-test on TWO RELATED samples of scores
    return t_test[1] #returns p-value only
#%%
'''
Calculate Search bias during probe
'''

def calc_time_in_area(target, array, idx_end, times, radius=15.):
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
    # x = target[0]
    # y = target[1]
    array = np.array(array[:idx_end], dtype=np.float64)

    # coords_in_range = ((array >= [x-radius, y-radius]) & (array <= [x+radius, y+radius])).all(axis=1) #gets list of coordinates within range (THIS IS A SQUARE)
    # idx_in_range = np.where(abs(coords_in_range)==True)[0] #gets index of the coordinates in range

    idx_in_range = np.where(abs(np.linalg.norm(target - array, ord=2, axis=1.) <= radius))[0] #gets array of coords in range (this is a circular range)
    
    #uses actual times from the experiments
    s = pd.Series(idx_in_range)
    time_in_range = np.sum([times[s.groupby(s.diff().gt(1).cumsum()).last().to_numpy()] - 
                            times[s.groupby(s.diff().gt(1).cumsum()).first().to_numpy()]])
    
    # time_in_range = len(idx_in_range)*0.0333333 #estimates time by multiplying samples with sampling rate (30fps)
    return time_in_range

#compares target dwell with REL targets, this only works with probe trials, so the center can be found
def compare_target_dwell(data, target, time_limit = '2min', radius = 15.):
    idx_end = pltlib.get_coords_timeLimit(data, time_limit)
    try: arena_center = data.arena_circle[:2]
    except: arena_center = pltlib.get_arena_center(data.r_nose)
    
    t1 = calc_time_in_area(target, data.r_nose, idx_end, data.time, radius)
    t2 = calc_time_in_area(pltlib.rotate(target, arena_center, 270), data.r_nose, idx_end, data.time, radius)
    t3 = calc_time_in_area(pltlib.rotate(target, arena_center, 180), data.r_nose, idx_end, data.time, radius)
    t4 = calc_time_in_area(pltlib.rotate(target, arena_center, 90), data.r_nose, idx_end, data.time, radius)
    return np.array([t1, t2, t3, t4])

#sums the dwell time of all probe trials in an experiment
def calc_search_bias(experiment, trial = 'Probe', time_limit = '2min', radius=15.):
    total_dwell = np.zeros(4)
    if type(experiment) is not list: experiment = [ experiment ] #checks if it is a list of experiments
    for exp in experiment:
        for files in os.listdir(glob.glob(glob.glob(PROCESSED_FILE_DIR+'/'+exp+'/')[0], recursive = True)[0]): #finds file path based on experiment
            if files.split('_')[-1].split('.')[0] == trial: #checks if it is the specified trial (default is probe trial)
                d = plib.TrialData()
                d.Load(exp, files.split('_')[-2].split('.')[0][1:], files.split('_')[-1].split('.')[0])
                total_dwell = total_dwell + compare_target_dwell(d, time_limit, radius)
                print('read mouse ' +files.split('_')[-2].split('.')[0][1:])
            
    return total_dwell

'''
Calculat spread along symmetry line betwen two points
'''

# def axis_of_symm(A, B): #gets axis of symmetry between two points
#     midpoint = ((A[0] + B[0])/2, (A[1] + B[1])/2)
#     theta = np.polyfit(np.array([A[0], B[0]]), np.array([A[1], B[1]]), 1) #gets line between two points
#     slope = (-1/theta[0]) #gets the perpendicular slope
#     b = midpoint[1] - (-1/theta[0]) * midpoint[0] #calculates perpendicular line bewteen points
#     return slope,  b

#line = a list of slope and b
def get_perp_intersect(point, line): #gets coords where point perpendicularly intersects with line
    point_m = -1/line[0]
    point_b = point[1]-(point_m*point[0])
    
    x = (point_b - line[1])/(line[0]-point_m)
    y = line[0]*x + line[1]
    return x, y


def dist_from_optimal_path(point, targetA, targetB): #calculates distance between point and perp intersect of point & line
    # optimal_path = np.polyfit(np.array([targetA[0], targetB[0]]), np.array([targetA[1], targetB[1]]), 1) #gets line between two points
    
    # line_intersect = get_perp_intersect(point, optimal_path)
    
    # distance = pltlib.calc_dist_bw_points(point, line_intersect)
    
    distance=np.cross(targetB-targetA,point-targetA)/np.linalg.norm(targetB-targetA) #inputs must be np arrays
    
    return distance

def calc_traj_spread(exp): #iterates over all points in the trial
    targetA = exp.target
    targetB = exp.target_reverse
    
    coord_interval = get_coords_bw_points(exp.r_nose, targetB, targetA)
    
    point_distances = []
    for c in coord_interval:
        point_distances.append(dist_from_optimal_path(c, targetA, targetB))

    max_distance = np.nanmax(np.abs(point_distances))
        
    return point_distances

def calc_spread_iterate_exp(experiments, show_load = True):
    #experiments is a list of trialdata experiments

    spreads = pd.DataFrame()
    distances = pd.DataFrame()
    distances_rand = pd.DataFrame()
    for t in experiments:
        spread =  calc_traj_spread(t) #calculates max distance from optimal line
        spread_max = np.nanmax(np.abs(spread))
        spread_avg = np.nanmean(np.abs(spread))
        spreads.at[t.trial, t.mouse_number] = spread_avg
        
        dist_AB = calc_dist_bw_points(t.r_center, t.target, t.target_reverse) #calculate distance travelled between two points
        distances.at[t.trial, t.mouse_number] = dist_AB
        
        if hasattr(t, 'arena_circle'):
            arena_center = t.arena_circle[:2]
        else: arena_center = pltlib.get_arena_center(t.r_nose)
        
        dist_rand = calc_dist_bw_points(t.r_center, pltlib.rotate(t.target, arena_center, 270), pltlib.rotate(t.target_reverse, arena_center, 270))
        distances_rand.at[t.trial, t.mouse_number] = dist_rand
    return {"Spreads": spreads, "Distance_AB": distances, "Distance_Random": distances_rand}

'''
Calculate vector angle
'''
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * (180 / np.pi)

# calculate difference in angle between two vectors in degrees
# def calc_angle_diff(a, b):
#     theta = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) * (180 / np.pi)
#     return theta

def vector_to_target(exp, coord_target, coord_end, coord_start = None):
    if coord_start is None: idx_start = 0
    else:
        idx_start = pltlib.coords_to_target(exp.r_nose, coord_start)
    idx_end = pltlib.coords_to_target(exp.r_nose, coord_end)
    
    # calculate mouse-to-target vector
    target_vector = coord_target - exp.r_center[idx_start:idx_end]
    target_direction = np.arctan2(target_vector[:,1],target_vector[:,0]) * (180 / np.pi)
    
    #calculate mouse-direction vector
    # mouse_vector = np.diff(exp.r_nose[idx_start:idx_end], axis=0)
    mouse_vector = exp.r_nose[idx_start:idx_end] - exp.r_center[idx_start:idx_end]
    mouse_direction = np.arctan2(mouse_vector[:,1],mouse_vector[:,0]) * (180 / np.pi)
    return target_vector, target_direction, mouse_vector, mouse_direction


# calculate mouse direction from nose point and center point until it reaches target
# should probably change this to calculate direction from movement vector
# mouse_vector = np.array([mat['r_nose'][:idx_end, 0] - mat['r_center'][:idx_end, 0],
#                          mat['r_nose'][:idx_end, 1] - mat['r_center'][:idx_end, 1]])
# mouse_direction = np.arctan(mouse_vector[0] / mouse_vector[1]) * (180 / np.pi)

def iterate_angle_difference(target_vector, mouse_vector):
    # calculate difference in angle between mouse direction and target direction
    angle_difference = []
    for i, _ in enumerate(mouse_vector):
        angle_difference.append(angle_between(target_vector[i], mouse_vector[i]))

    return angle_difference
#%%
if __name__ == '__main__':
    
    exp = plib.TrialData()
    exp.Load('2019-10-07', '16', 'Probe')
    dwells = compare_target_dwell(exp)
    
    
    
