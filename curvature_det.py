# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 22:55:32 2023

@author: Kelly
"""
import numpy as np

def find_trajectory_hole_intersections(data, r_holes, idx_end = False, hole_radius = 5.):
    '''
    finds when mouse trajectory intersects with a hole within a certain radius

    Parameters
    ----------
    data : class
        class containing trial information such as x y coordinates.
    r_holes : array of float
        array of tuples representing the coordinate locations of all the holes.
    idx_end : int, optional
        index where we want to crop the trajectory. If false, the trajectory is not cropped. The default is False.
    hole_radius : float, optional
        distance from the hole where detection should occur. The default is 5.

    Returns
    -------
    idx_inter : list of arrays
        list of all holes by key. If intersection occured, the corresponding array will contain the indices of the trajectory that intersect with the hole

    '''
    
    if isinstance(idx_end, bool) or idx_end > len(data.r_center): #check if we are cropping trajectory
        idx_end = len(data.r_center)
    
    if len(data.r_nose) > 1.: bodypoint = data.r_nose #try to use the nose points if they exist
    else: bodypoint = data.r_center
    
    idx_inter = [] #list of indices where intersection happens (one arena hole per list item)
    
    for i,r0 in enumerate(r_holes): # counting the keys and coords for each hole
        tind = np.nonzero(np.linalg.norm(bodypoint[:idx_end,:] - r0,axis=1) < hole_radius)[0] #linalg.norm calculates magnitude vector between coords and holes
        if len(tind) > 0: #intersection(s) occured
            idx_inter.append(tind)
        else: #intersections not found
            idx_inter.append(None)
    return idx_inter

def curvature(A,B,C):
    """Calculates the Menger curvature from three Points, given as numpy arrays.
    Source: https://stackoverflow.com/questions/55526575/python-automatic-resampling-of-data
    """

    # Pre-check: Making sure that the input points are all numpy arrays
    if any(x is not np.ndarray for x in [type(A),type(B),type(C)]):
        print("The input points need to be a numpy array, currently it is a ", type(A))

    # Augment Columns
    A_aug = np.append(A,1)
    B_aug = np.append(B,1)
    C_aug = np.append(C,1)

    # Caclulate Area of Triangle
    matrix = np.column_stack((A_aug,B_aug,C_aug))
    area = 1/2*np.linalg.det(matrix)

    # Special case: Two or more points are equal 
    if np.all(A == B) or  np.all(B == C) or np.all(A == C):
        curvature = 0
    else:
        curvature = 4*area/(np.linalg.norm(A-B)*np.linalg.norm(B-C)*np.linalg.norm(C-A))

    # Return Menger curvature
    return curvature

def get_traj_curvatures(data, idx_end):
    '''calculates a list of the trajectory's' curvatures'''
    try: x, y = data.r_nose[:idx_end,0], data.r_nose[:idx_end,1] #try to use nose points if they exist
    except: x, y = data.r_center[:idx_end,0], data.r_center[:idx_end,1]
    
    curvature_list = np.empty(0) 
    for i in range(len(x)-2):
        # Get the three points
        A = np.array([x[i],y[i]])
        B = np.array([x[i+1],y[i+1]])
        C = np.array([x[i+2],y[i+2]])
    
        # Calculate the curvature
        curvature_value = abs(curvature(A,B,C))
        curvature_list = np.append(curvature_list, curvature_value)
    curvature_list = np.append(curvature_list, [0,0]) #add two trailing zeros so it is the same length as original coords
    return curvature_list

def peakdet(v, delta, x = None):
    """
    Made by endolith at https://gist.github.com/endolith/250860
    Based on MATLAB script by Eli Billauer at http://billauer.co.il/peakdet.html
    
    Returns two arrays, peaks and valleys
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        raise ValueError('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        raise ValueError('Input argument delta must be a scalar')
    
    if delta <= 0:
        raise ValueError('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)

def find_sharp_curve_near_hole(curvatures, idx_inter, delta = 1.):
    '''
    calculates when the mouse curves sharply near a hole

    Parameters
    ----------
    curvatures : array of float64
        A Nx1 array of the curvature values along the trajectory path
    idx_inter : list
        list of arrays which contains the indexes of when the mouse intersects with a hole. 
        The index of the list corresponds to the hole index
    delta : float
        A parameter for the peakdet function. The default is 1..

    Returns
    -------
    peaks : array of int32
        array of all the peaks detected. Can be graphed for verification reasons.
    idx_traj_holes_curve : list
        index to get timepoints when curving occured in vicinity of hole.

    '''
   
    peaks, _ = peakdet(curvatures, delta) #we only need the peaks, usually delta = 1
    
    # f = (velocity[:-1]-threshold)*(velocity[1:]-threshold) # f<=0 -> crossing of velocity threshold;
    # idx_traj_slow = np.nonzero(f <= 0)[0] # index of all threshold crossings
    # idx_traj_slow = np.nonzero(np.logical_and( f <= 0 , velocity[1:]<=threshold))[0] #index of only downward crossings
    
    #try filtering peaks with idx_traj_slow to combine curvature and speed detection
    idx_traj_holes_curve = [] #index to get timepoints when curving occured in vicinity of hole
    for i,k in enumerate(idx_inter):
        if k is not None:
            if len(np.intersect1d(peaks, k)) > 0:
                idx_traj_holes_curve.append(np.intersect1d(peaks, k))
            else: #intersected hole but didn't check
                idx_traj_holes_curve.append(None)
        else: #didn't go to hole
            idx_traj_holes_curve.append(None)

    return peaks[:,0].astype(int), idx_traj_holes_curve


def get_times(data, idx_inter, idx_traj_holes_curve):
    '''
    returns timepoints when hole checking occured. 
    times = timepoints in seconds. 
    k_times = timepoints in index
    '''
    already_visited_holes = []
    k_times = [] # hole key and trajectory index
    for i, hole in enumerate(idx_inter):
        if hole is not None and idx_traj_holes_curve[i] is not None:
            for x in enumerate(hole):
                if x[0]-x[1] in already_visited_holes: #did we already group this sequence?
                    pass
                elif np.intersect1d(idx_traj_holes_curve[i], x[1]).size != 0: #does this group intersect with hole checking?
                    for h in idx_traj_holes_curve[i]:
                        if h == x[1]: 
                            k_times.append([i,x[1]]) #append both hole key and trajectory index
                            already_visited_holes.append(x[0]-x[1])
    k_times=np.array(k_times)
                
    
    times = [] #actual timepoints in seconds
    for i in k_times:
        times.append(data.time[i[1]])
    times = sorted(list(set(times))) #remove duplicates & sorts in decending order
    return times, k_times