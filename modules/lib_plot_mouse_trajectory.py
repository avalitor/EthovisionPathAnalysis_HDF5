# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:04:44 2022

functions for plotting search trajectories

@author: Kelly
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg #for plotting the image
import matplotlib.colors as mcolors #for heatmap colours
import matplotlib.patches as patches #to crop image
from scipy.ndimage.filters import gaussian_filter #for heatmap
from modules import lib_process_data_to_mat as plib
from modules import lib_experiment_parameters as params
from modules.config import ROOT_DIR

'''
Helper Functions
'''

#calculate distance between two points
def calc_dist_bw_points(coord1, coord2):
    dist = np.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)
    return dist

#finds index of first coordinate within a certain distance of target
def coords_to_target(coords, target):

    for i in coords:
        if calc_dist_bw_points(i, target) <= 5.:
            idx_end = np.where(coords==i)[0][0]
            break
        else: idx_end = len(coords)-2 #minus 2 so it is in bounds of the array
        
    if idx_end == len(coords)-2: print('WARNING ::: Target Never Reached')
    
    # cropcoords = coords[:index+1] #add one to include the indexed coordinate
    return idx_end

#gets starting index of last continuous search trajectory before reaching target
def continuous_coords_to_target(d, idx_end):
    all_points = np.concatenate((d.r_nose, d.r_center, d.r_tail), axis=1) #combine all coordinate arrays together
    all_points_crop = all_points[:idx_end] #crop so it ends at target
    bool_arr = np.isnan(all_points_crop).all(axis=1) #finds where the row is all nan
    nan_idx = np.where(bool_arr) #returns indexes where the rows were nan
    if len(nan_idx[0]) > 0 and bool_arr[-1] != True: #checks if any nans were found and if the target was reached
        idx_start = nan_idx[0][-1] #gets last value
    else: idx_start = 0
    return idx_start

#rotate coordinates (Nx2 array) about a center point (tuple) in degrees
def rotate(coords, center, degrees):
    angle = np.deg2rad(degrees) #convert to radians
    return np.dot(coords-center,np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]))+center

#gets the center of a set of coordinates, useful for coordinate rotation but only if coordinates cover the full extent of the arena
def get_arena_center(coords):
    x = coords[:,0]
    y = coords[:,1]
    x_center = np.nanmax(x)-((np.nanmax(x)-np.nanmin(x))/2)
    y_center = np.nanmax(y)-((np.nanmax(y)-np.nanmin(y))/2)
    return (x_center,y_center)

def get_coords_timeLimit(trial, time_limit):
    if time_limit == 'all':
        idx_end = len(trial.time[0])
    if time_limit == '2min':
        idx_end = np.searchsorted(trial.time[0], 120.)
    if time_limit == '5min':
        idx_end = np.searchsorted(trial.time[0], 300.)
        
    # x = trial.r_nose[:idx_end,0]
    # y = trial.r_nose[:idx_end,1]
    return idx_end

def make_heatmap(x, y, s, bins=1000):
    #gets rid of the NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

#%%
'''
Plotting Functions
'''

def plot_single_traj(trialclass, cropcoords = True, crop_end_custom = False, crop_interval = False, returnpath = False, continuous = False, savefig = False):
    '''
    Plots r_nose coordinates on an image based on data in the class TrialData
    
    Parameters
    ----------
    trialclass : class
        Object class that contains all trial info
    cropcoords: bool, optional
        crops the trajectory when the mouse reaches the target
    crop_end_custom: bool or tuple, optional
        crops the trajectory at a custom coordinate
    crop_interval: bool or array of two tuples, optional
        crops the trajectory between two coordinates
    returnpath : bool, optional
        Plot the return path from the target. The default is False.
    continuous : bool, optional
        Plots the last continuous trajectory to target
    savefig : book, optional
        Saves the figure. The default is False.
    '''

    fig, ax = plt.subplots()

    #import image
    img = mpimg.imread(os.path.join(ROOT_DIR, 'data', 'BackgroundImage', trialclass.bkgd_img))
    ax.imshow(img, extent=trialclass.img_extent) #plot image to match ethovision coordinates
    
    line_colour = '#004E89'
    
    if cropcoords == True and isinstance(crop_end_custom, bool) and isinstance(crop_interval, bool): #if we want to crop at target and not at custom location or interval
        #get target index
        index = coords_to_target(trialclass.r_nose, trialclass.target)
        
        #gets starting index if plotting continuous trajectory
        if continuous == True: idx_start = continuous_coords_to_target(trialclass, index)
        else: idx_start = 0
            
        #plot path to target
        ax.plot(trialclass.r_nose[idx_start:index+1,0], trialclass.r_nose[idx_start:index+1,1], ls='-', color = line_colour)
    elif isinstance(crop_end_custom, bool) == False: #checks if we want to crop at specific coordinate
        #get target index
        index = coords_to_target(trialclass.r_nose, crop_end_custom)
        
        # #if cropcoords is also true, crops at whichever one comes first
        # if cropcoords == True: 
        #     index2 = coords_to_target(trialclass.r_nose, trialclass.target)
        #     if index2 <= index: index = index2
                
        #plot path to custom target
        ax.plot(trialclass.r_nose[:index+1,0], trialclass.r_nose[:index+1,1], ls='-', color = line_colour)
        
    elif isinstance(crop_interval, bool) == False: #check if we want to crop trajectory between two points
        #get start and end index
        idx_start = coords_to_target(trialclass.r_nose, crop_interval[0])
        idx_end = coords_to_target(trialclass.r_nose, crop_interval[1])
        #plot path between coordinates
        ax.plot(trialclass.r_nose[idx_start:idx_end+1,0], trialclass.r_nose[idx_start:idx_end+1,1], ls='-', color = line_colour)
            
    else: 
        ax.plot(trialclass.r_nose[:,0], trialclass.r_nose[:,1], ls='-', color = line_colour)
    #plot return path
    if returnpath == True:
        ax.plot(trialclass.r_nose[index:,0], trialclass.r_nose[index:,1], ls='-', color = '#717171')

    #plot path with colours
#        N = np.linspace(0, 10, np.size(y))
#        ax.scatter(x, y, s=1.5, c = N, cmap=cm.jet_r, edgecolor='none')

    #annotate image
    target = plt.Circle((trialclass.target), 2.5, color='b')
    ax.add_artist(target)
    
    # test = plt.Circle((-37.48, -11.65), 2.5, color='orange')
    # ax.add_artist(test)
    
    if params.check_reverse(trialclass.exp, trialclass.trial) is True: #annotates false target, optional
        prev_target = plt.Circle((params.set_reverse_target(trialclass.exp, trialclass.entrance, trialclass.trial)), 2.5, color='r')
        ax.add_artist(prev_target)

    # plt.style.use('default')
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if savefig == True:
        plt.savefig(ROOT_DIR+'/figures/Plot_%s_M%s_%s.png'%(trialclass.protocol_name, trialclass.mouse_number, trialclass.trial), dpi=600, bbox_inches='tight', pad_inches = 0)
        
    plt.show()
    
def plot_multi_traj(trialclass_list, crop_rev = False, savefig = False):
    '''
    Plots multiple trajectories on a single image
    If the targets rotate with the entrances, automatically align the trajectories so the targets are the same

    Parameters
    ----------
    trialclass_list : list
        List of class TrialData
    crop_rev: bool, optional
        crops trajectory at reverse target, if it exists
    savefig : bool, optional
        Save this figure as a png image. The default is False.
    '''

    fig, ax = plt.subplots()

    #import image
    img = mpimg.imread(os.path.join(ROOT_DIR, 'data', 'BackgroundImage', trialclass_list[0].bkgd_img))
    ax.imshow(img, extent=trialclass_list[0].img_extent) #plot image to match ethovision coordinates
    
    if all(trialclass_list[0].target != trialclass_list[1].target): #if the targets change between trials
        temp = plib.TrialData()
        temp.Load(trialclass_list[0].exp, '*', 'Probe')
        origin = get_arena_center(temp.r_nose) #get center of rotation
        for t in trialclass_list: #rotate all coordinates so they align
            if t.entrance == 'SE': t.r_nose_r = rotate(t.r_nose, origin, 270)
            elif t.entrance == 'NE': t.r_nose_r = rotate(t.r_nose, origin, 180)
            elif t.entrance == 'NW': t.r_nose_r = rotate(t.r_nose, origin, 90)
            else: t.r_nose_r = t.r_nose
    
    colours = iter(['#004E89', '#C00021', '#5F0F40', '#F18701', '#FFD500'])
    # linestyles = iter(['-', '--', ':'])
    for t in trialclass_list:
        if len(t.time)>0: #checks to see if list data is not empty
            
            if crop_rev == True:
                try : index = coords_to_target(t.r_nose, t.target_reverse)
                except:index = coords_to_target(t.r_nose, t.target)
            else: index = coords_to_target(t.r_nose, t.target)

            #plot path to target
            try: ax.plot(t.r_nose_r[:index+1,0], t.r_nose_r[:index+1,1], ls='-', color= next(colours, 'k'), alpha=1.) #iterates over colours until it ends at black, next(colours, 'k')
            except: ax.plot(t.r_nose[:index+1,0], t.r_nose[:index+1,1], ls='-', color= next(colours, 'k'), alpha=1.) #iterates over colours until it ends at black, next(colours, 'k')
        

    #annotate image
    target = plt.Circle((trialclass_list[0].target), 2.5, color='b')
    ax.add_artist(target)
    

    # plt.style.use('default')
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if savefig == True:
        plt.savefig(ROOT_DIR+'/figures/Plot_%s_Trial%s-%s_Combo.png'%(trialclass_list[0].protocol_name, trialclass_list[0].trial, trialclass_list[1].trial), 
                    dpi=600, bbox_inches='tight', pad_inches = 0)
        
    plt.show()

def plot_heatmap(trialclass_list, time_limit = '2min', savefig = False):
    
    
    if type(trialclass_list) is list: #checks if plotting multiple trials or just one
        first_trial = trialclass_list[0]

        if all(first_trial.target != trialclass_list[1].target): #if the targets change between trials
            temp = plib.TrialData()
            temp.Load(first_trial.exp, '*', 'Probe')
            origin = get_arena_center(temp.r_nose) #get center of rotation
            for t in trialclass_list: #rotate all coordinates so they align
                if t.entrance == 'SE': t.r_nose_r = rotate(t.r_nose, origin, 270)
                elif t.entrance == 'NE': t.r_nose_r = rotate(t.r_nose, origin, 180)
                elif t.entrance == 'NW': t.r_nose_r = rotate(t.r_nose, origin, 90)
                else: t.r_nose_r = t.r_nose
        
        
            if all(first_trial.target != trialclass_list[-1].target): #if target changes between experiments
                temp = plib.TrialData()
                temp.Load(trialclass_list[-1].exp, '*', 'Probe')
                origin = get_arena_center(temp.r_nose) #get center of rotation
    
                for t in trialclass_list:
                    if len(t.time)>0: #checks to see if list data is not empty
                        if t.exp == u'2021-11-15': 
                            t.r_nose_r = rotate(t.r_nose, origin, 180) #so it matches experiment jun16
                            print(t.mouse + t.trial)
                        elif t.exp == u'2021-08-11':
                            t.r_nose_r = rotate(t.r_nose, origin, 90) #so it matches experiment 2019-12-11 **NOT WORKING**
                            print(t.mouse + t.trial)
                        else: t.r_nose_r = t.r_nose
        
        combocoords = np.empty([0,2])
        for t in trialclass_list:
            if len(t.time)>0: #checks to see if list data is not empty
                #get time limit index and append to array

                try: combocoords = np.concatenate((combocoords, t.r_nose_r[:get_coords_timeLimit(t, time_limit)+1,:]), axis = 0)
                except: combocoords = np.concatenate((combocoords, t.r_nose[:get_coords_timeLimit(t, time_limit)+1,:]), axis = 0)
                x = combocoords[:,0]
                y = combocoords[:,1]

                
    else: 
        first_trial = trialclass_list
        x = trialclass_list.r_nose[:get_coords_timeLimit(trialclass_list, time_limit),0]
        y = trialclass_list.r_nose[:get_coords_timeLimit(trialclass_list, time_limit),1]
    
    
    fig, ax = plt.subplots()
    
    #import image
    img = mpimg.imread(os.path.join(ROOT_DIR, 'data', 'BackgroundImage', first_trial.bkgd_img))
    im = ax.imshow(img, extent=first_trial.img_extent) #plot image to match ethovision coordinates
    
    #crops the image to 130% of coordinate limits
    patch = patches.Circle(get_arena_center(first_trial.r_nose), 
                           radius=((np.nanmax(x)-np.nanmin(x))/2)*1.30, 
                           transform=ax.transData)
    im.set_clip_path(patch)
    
    #plot path
    # ax.plot(x, y, ls='-', color = 'red')
    
    #plot heatmap
    img, extent = make_heatmap(x, y, 32)
    colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=20)
    hm = ax.imshow(img, extent=extent, origin='lower', cmap=cmapred) #othor colours: rainbow_alpha, cm.jet
    plt.colorbar(mappable=hm)
    
    
    
    #scales the heatmap to match the background picture
    plt.xlim(first_trial.img_extent[0], first_trial.img_extent[1]) 
    plt.ylim(first_trial.img_extent[2], first_trial.img_extent[3])
    
    # Remove ticks
    ax.axis('off')
    
    if savefig == True:
        if type(trialclass_list) is list:
            plt.savefig(ROOT_DIR+'/figures/Heatmap_%s_M%s-%s.png'%(first_trial.protocol_name, first_trial.mouse_number, trialclass_list[-1].mouse_number), dpi=600, bbox_inches='tight', pad_inches = 0)
        else:
            plt.savefig(ROOT_DIR+'/figures/Heatmap_%s_M%s_%s.png'%(first_trial.protocol_name, first_trial.mouse_number, first_trial.trial), dpi=600, bbox_inches='tight', pad_inches = 0)
     
    plt.show()
    
    
def plot_2_target_analysis(trialclass, cropcoords = True, crop_end_custom = False, crop_interval = False, returnpath = False, continuous = False, savefig = False):
    '''
    Plots r_nose coordinates on an image based on data in the class TrialData
    
    Parameters
    ----------
    trialclass : class
        Object class that contains all trial info
    cropcoords: bool, optional
        crops the trajectory when the mouse reaches the target
    crop_end_custom: bool or tuple, optional
        crops the trajectory at a custom coordinate
    crop_interval: bool or array of two tuples, optional
        crops the trajectory between two coordinates
    returnpath : bool, optional
        Plot the return path from the target. The default is False.
    continuous : bool, optional
        Plots the last continuous trajectory to target
    savefig : book, optional
        Saves the figure. The default is False.
    '''

    fig, ax = plt.subplots()

    #import image
    img = mpimg.imread(os.path.join(ROOT_DIR, 'data', 'BackgroundImage', trialclass.bkgd_img))
    ax.imshow(img, extent=trialclass.img_extent) #plot image to match ethovision coordinates
    
    if cropcoords == True and isinstance(crop_end_custom, bool) and isinstance(crop_interval, bool): #if we want to crop at target and not at custom location or interval
        #get target index
        index = coords_to_target(trialclass.r_nose, trialclass.target)
        
        #gets starting index if plotting continuous trajectory
        if continuous == True: idx_start = continuous_coords_to_target(trialclass, index)
        else: idx_start = 0
            
        #plot path to target
        ax.plot(trialclass.r_nose[idx_start:index+1,0], trialclass.r_nose[idx_start:index+1,1], ls='-', color = 'k')
    elif isinstance(crop_end_custom, bool) == False: #checks if we want to crop at specific coordinate
        #get target index
        index = coords_to_target(trialclass.r_nose, crop_end_custom)
        #plot path to target
        ax.plot(trialclass.r_nose[:index+1,0], trialclass.r_nose[:index+1,1], ls='-', color = 'k')
    elif isinstance(crop_interval, bool) == False: #check if we want to crop trajectory between two points
        #get start and end index
        idx_start = coords_to_target(trialclass.r_nose, crop_interval[0])
        idx_end = coords_to_target(trialclass.r_nose, crop_interval[1])
        #plot path between coordinates
        ax.plot(trialclass.r_nose[idx_start:idx_end+1,0], trialclass.r_nose[idx_start:idx_end+1,1], ls='-', color = 'k')
            
    else: 
        ax.plot(trialclass.r_nose[:,0], trialclass.r_nose[:,1], ls='-', color = 'k')
    #plot return path
    if returnpath == True:
        ax.plot(trialclass.r_nose[index:,0], trialclass.r_nose[index:,1], ls='-', color = '#717171')

    #plot path with colours
#        N = np.linspace(0, 10, np.size(y))
#        ax.scatter(x, y, s=1.5, c = N, cmap=cm.jet_r, edgecolor='none')

    #annotate image
    target = plt.Circle((trialclass.target), 2.5, color='b')
    ax.add_artist(target)
    
    test2 = (0.5129473857826159, 7.744444169553028)


    x = np.arange(-150, 150)
    ax.plot(x, test2[0]*x+test2[1], ls='-', color = 'k') #line of symmetry
    
    # ax.plot(test[:,0], test[:,1])
    
    if params.check_reverse(trialclass.exp, trialclass.trial) is True: #annotates false target, optional
        prev_target = plt.Circle((params.set_reverse_target(trialclass.exp, trialclass.entrance, trialclass.trial)), 2.5, color='r')
        ax.add_artist(prev_target)

    # plt.style.use('default')
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if savefig == True:
        plt.savefig(ROOT_DIR+'/figures/Plot_%s_M%s_%s.png'%(trialclass.protocol_name, trialclass.mouse_number, trialclass.trial), dpi=600, bbox_inches='tight', pad_inches = 0)
        
    plt.show()

if __name__ == '__main__': #only runs this function if the script top level AKA is running by itself
    exp = plib.TrialData()
    exp.Load('2022-08-12', '72', '22')
    print('Mouse %s Trial %s'%(exp.mouse_number, exp.trial))

    plot_single_traj(exp, cropcoords = True)
    pass