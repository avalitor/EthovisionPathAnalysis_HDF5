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
from scipy.ndimage import gaussian_filter #for heatmap
from modules import lib_process_data_to_mat as plib
from modules import lib_experiment_parameters as params
from modules.config import ROOT_DIR
import modules.calc_latency_distance_speed as calc

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
        if calc_dist_bw_points(i, target) <= 3.: #try 1.0 or 3.0
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
    if np.isnan(coords).all():
        x_center, y_center = 0,0
        print("WARNING: One or more trials are empty")
    else:
        x = coords[:,0]
        y = coords[:,1]
        x_center = np.nanmax(x)-((np.nanmax(x)-np.nanmin(x))/2)
        y_center = np.nanmax(y)-((np.nanmax(y)-np.nanmin(y))/2)
    return (x_center,y_center)

def get_coords_timeLimit(trial, time_limit):
    assert time_limit in ['all','2min','5min'], "time_limit must be 'all', '2min','5min'"
    if time_limit == 'all':
        idx_end = len(trial.time)
    if time_limit == '2min':
        idx_end = np.searchsorted(trial.time, 120.)
    if time_limit == '5min':
        idx_end = np.searchsorted(trial.time, 300.)
        
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
'''helper plotting functions'''
def draw_arena(data, ax):
    #draws arena
    Drawing_arena_circle = plt.Circle( (data.arena_circle[0], data.arena_circle[1]), 
                                          data.arena_circle[2] , fill = False )
    ax.add_artist( Drawing_arena_circle )
    
    for c in data.r_arena_holes:
        small_hole = plt.Circle( (c[0], c[1] ), 0.5 , fill = False ,alpha=0.5)
        ax.add_artist( small_hole )
        
        ax.set_aspect('equal','box')
        ax.set_xlim([data.img_extent[2],data.img_extent[3]])
        ax.set_ylim([data.img_extent[2],data.img_extent[3]])
        ax.axis('off')
    return ax

#%%
'''
Main Plotting Functions
'''

def plot_single_traj(trialclass, show_target = True, cropcoords = True, crop_end_custom = False, crop_interval = False, 
                     returnpath = False, continuous = False, savefig = False):
    '''
    Plots r_nose coordinates on an image based on data in the class TrialData
    
    Parameters
    ----------
    trialclass : class
        Object class that contains all trial info
    show_target : bool, optional
        draw target circle
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
    

    if hasattr(trialclass, 'arena_circle'):
        # #crops the image to 130% of coordinate limits
        # patch = patches.Circle(trialclass_list[0].arena_circle[:2], 
        #                        radius=(trialclass_list[0].arena_circle[2]*1.3), 
        #                        transform=ax.transData)
        # im.set_clip_path(patch)
        draw_arena(trialclass, ax)
    else: 
        print('Missing arena circle coordinates')
        #import image
        img = mpimg.imread(os.path.join(ROOT_DIR, 'data', 'BackgroundImage', trialclass.bkgd_img))
        im = ax.imshow(img, extent=trialclass.img_extent) #plot image to match ethovision coordinates
    
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
        
        #gets starting index if plotting continuous trajectory
        if continuous == True: idx_start = continuous_coords_to_target(trialclass, index)
        else: idx_start = 0
                
        #plot path to custom target
        ax.plot(trialclass.r_nose[idx_start:index+1,0], trialclass.r_nose[idx_start:index+1,1], ls='-', color = line_colour)
        
    elif isinstance(crop_interval, bool) == False: #check if we want to crop trajectory between two points
        #get start and end index
        idx_start = coords_to_target(trialclass.r_nose, crop_interval[0])
        idx_end = coords_to_target(trialclass.r_nose, crop_interval[1])
        if idx_start > idx_end: #reverse them if the mouse arrives at the second index before the first
            idx_start = coords_to_target(trialclass.r_nose, crop_interval[1])
            idx_end = coords_to_target(trialclass.r_nose, crop_interval[0])
        #plot path between coordinates
        ax.plot(trialclass.r_nose[idx_start:idx_end+1,0], trialclass.r_nose[idx_start:idx_end+1,1], ls='-', color = line_colour)
            
        #annotate the two points
        point1 = plt.Circle((crop_interval[0]), 2.5, color='#F18701')
        point2 = plt.Circle((crop_interval[1]), 2.5, color='#F18701')
        ax.add_artist(point1)
        ax.add_artist(point2)
    else: 
        ax.plot(trialclass.r_nose[:,0], trialclass.r_nose[:,1], ls='-', color = line_colour)
    #plot return path
    if returnpath == True:
        ax.plot(trialclass.r_nose[index:,0], trialclass.r_nose[index:,1], ls='-', color = '#717171')

    #plot path with colours
#        N = np.linspace(0, 10, np.size(y))
#        ax.scatter(x, y, s=1.5, c = N, cmap=cm.jet_r, edgecolor='none')

    #annotate image
    if show_target == True:
        target = plt.Circle((trialclass.target), 2.5, color='b')
        ax.add_artist(target)
    
    
    if params.check_reverse(trialclass.exp, trialclass.trial) is True: #annotates false target, optional
        prev_target = plt.Circle((params.set_reverse_target(trialclass.exp, trialclass.entrance, trialclass.trial)), 2.5, color='r')
        ax.add_artist(prev_target)

    # plt.style.use('default')
    
    #draw entrance
    for i, _ in enumerate(trialclass.r_nose):
        if np.isnan(trialclass.r_nose[i][0]): continue
        else:
            first_coord = trialclass.r_nose[i]
            break
    entrance = plt.Rectangle((first_coord-3.5), 7, 7, fill=False, color='k', alpha=0.8, lw=3)
    ax.add_artist(entrance)   
    
    ax.axis('off') #remove border

    if savefig == True:
        plt.savefig(ROOT_DIR+'/figures/Plot_%s_M%s_%s.png'%(trialclass.protocol_name, trialclass.mouse_number, trialclass.trial), dpi=600, bbox_inches='tight', pad_inches = 0)
        
    plt.show()
    
def plot_multi_traj(trialclass_list, align_entrance = True, crop_target = False, crop_rev = False, crop_end_custom = False, crop_interval = False, continuous = False, savefig = False):
    '''
    Plots multiple trajectories on a single image
    If the targets rotate with the entrances, automatically align the trajectories so the targets are the same

    Parameters
    ----------
    trialclass_list : list
        List of class TrialData
    align_entrance : bool, optional
        aligns all rotating entrances to the same entrance
    crop_target: bool
        crops each trajectory at target
    crop_rev: bool, optional
        crops trajectory at reverse target, if it exists
    crop_end_custom : bool or list of tuples, optional
        crops each trajectory at custom coordinate
    crop_interval: bool or list of tuples, optional
        crops the trajectory between two coordinates
    savefig : bool, optional
        Save this figure as a png image. The default is False.
    '''

    fig, ax = plt.subplots()

    if hasattr(trialclass_list[0], 'arena_circle'):
        # #crops the image to 130% of coordinate limits
        # patch = patches.Circle(trialclass_list[0].arena_circle[:2], 
        #                        radius=(trialclass_list[0].arena_circle[2]*1.3), 
        #                        transform=ax.transData)
        # im.set_clip_path(patch)
        draw_arena(trialclass_list[0], ax)
    else: 
        print('Missing arena circle coordinates')
        #import image
        img = mpimg.imread(os.path.join(ROOT_DIR, 'data', 'BackgroundImage', trialclass_list[0].bkgd_img))
        im = ax.imshow(img, extent=trialclass_list[0].img_extent) #plot image to match ethovision coordinates
        

    if align_entrance:
        if all(trialclass_list[0].target != trialclass_list[1].target): #if the targets change between trials
            temp = plib.TrialData()
            temp.Load(trialclass_list[0].exp, '*', 'Probe')
            origin = get_arena_center(temp.r_nose) #get center of rotation
            for t in trialclass_list: #rotate all coordinates so they align
                if hasattr(t, 'arena_circle'): origin = t.arena_circle[:2]
                if t.entrance == 'SE': t.r_nose_r = rotate(t.r_nose, origin, 270)
                elif t.entrance == 'NE': t.r_nose_r = rotate(t.r_nose, origin, 180)
                elif t.entrance == 'NW': t.r_nose_r = rotate(t.r_nose, origin, 90)
                # else: t.r_nose_r = t.r_nose
    
    # colours = iter(['#004E89', '#C00021', '#5F0F40', '#F18701', '#FFD500'])
    colours = iter(['k', '#C00021', '#004E89', '#F18701', '#FFD500'])
    alpha = iter([0.3, 1, 1])
    # linestyles = iter(['-', '--', ':'])
    if isinstance(crop_end_custom, bool) == False: #checks to see if we are using custom crop coordinates
        if type(crop_end_custom) is not list: print('Error: crop custom not a list') #checks to make sure it is a list
        else: target_custom = iter(crop_end_custom) #sets up an iterating coordinate
    if isinstance(crop_interval, bool) == False:
        if type(crop_interval) is not list: print('Error: crop custom not a list') #checks to make sure it is a list
        else: 
            targets_custom = iter(crop_interval)
            
    
    for t in trialclass_list:
        if len(t.time)==0: #checks to see if list data is not empty
            raise ValueError("the list contains empty data")
        
        if crop_rev == True:
            try : index = coords_to_target(t.r_nose, t.target_reverse)
            except:index = coords_to_target(t.r_nose, t.target)
        elif isinstance(crop_end_custom, bool) == False:
            index = coords_to_target(t.r_nose, next(target_custom))
        elif isinstance(crop_interval, bool) == False: #check if we want to crop trajectory between two points
            #get start and end index
            target_coords = next(targets_custom)
            idx_start = coords_to_target(t.r_nose, target_coords[0])
            index = coords_to_target(t.r_nose, target_coords[1])
            if idx_start > index: #reverse them if the mouse arrives at the second index before the first
                temp = idx_start
                idx_start = index
                index = temp
            if hasattr(t, 'r_nose_r'): pass
            else:
                #annotate the two points
                point1 = plt.Circle((target_coords[0]), 2.5, color='b', zorder=10)
                point2 = plt.Circle((target_coords[1]), 2.5, color='b', zorder=10)
                ax.add_artist(point1)
                ax.add_artist(point2)
            
        elif crop_target: index = coords_to_target(t.r_nose, t.target)
        else: index = len(t.r_nose)
        
        #gets starting index if plotting continuous trajectory
        if continuous == True: idx_start = continuous_coords_to_target(t, index)
        elif isinstance(crop_interval, bool) == True: idx_start = 0

        #plot path to target
        try: ax.plot(t.r_nose_r[idx_start:index+1,0], t.r_nose_r[idx_start:index+1,1], ls='-', color= next(colours, 'k'), alpha=next(alpha, 1)) #iterates over colours until it ends at black, next(colours, 'k')
        except: ax.plot(t.r_nose[idx_start:index+1,0], t.r_nose[idx_start:index+1,1], ls='-', color= next(colours, 'k'), alpha=next(alpha, 1)) #iterates over colours until it ends at black, next(colours, 'k')
    

    #annotate image
    if crop_rev:
        target = plt.Circle((trialclass_list[0].target_reverse), 2.5, color='b')
        ax.add_artist(target)
    elif crop_end_custom:
        target = plt.Circle((crop_end_custom), 2.5, color='b')
        ax.add_artist(target)
    elif crop_target:
        target = plt.Circle((trialclass_list[0].target), 2.5, color='g')
        ax.add_artist(target)
    
    

    # plt.style.use('default')
    ax.axis('off') #remove border

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
                if hasattr(t, 'arena_circle'): origin = t.arena_circle[:2]
                if t.entrance == 'SE': t.r_nose_r = rotate(t.r_nose, origin, 270)
                elif t.entrance == 'NE': t.r_nose_r = rotate(t.r_nose, origin, 180)
                elif t.entrance == 'NW': t.r_nose_r = rotate(t.r_nose, origin, 90)
                else: t.r_nose_r = t.r_nose
        
        
            if all(first_trial.target != trialclass_list[-1].target): #if target changes between experiments
                temp = plib.TrialData()
                temp.Load(trialclass_list[-1].exp, '*', 'Probe')
                origin = get_arena_center(temp.r_nose) #get center of rotation
                if hasattr(t, 'arena_circle'): origin = t.arena_circle[:2]
    
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
    
    if hasattr(first_trial, 'arena_circle'):
        #crops the image to 130% of coordinate limits
        patch = patches.Circle(first_trial.arena_circle[:2], 
                               radius=(first_trial.arena_circle[2]*1.3), 
                               transform=ax.transData)
        im.set_clip_path(patch)
    else: print('Missing arena circle coordinates')
    # #crops the image to 130% of coordinate limits
    # patch = patches.Circle(get_arena_center(first_trial.r_nose), 
    #                        radius=((np.nanmax(x)-np.nanmin(x))/2)*1.30, 
    #                        transform=ax.transData)
    # im.set_clip_path(patch)
    
    #plot path
    # ax.plot(x, y, ls='-', color = 'red')
    
    #plot heatmap
    img, extent = make_heatmap(x, y, 32)
    colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=20)
    hm = ax.imshow(img, extent=extent, origin='lower', cmap=cmapred) #othor colours: rainbow_alpha, cm.jet
    # plt.colorbar(mappable=hm) #add colorbar
    
    #plot target
    target = plt.Circle((first_trial.target), radius = 15., fill = False, ec = 'g', linestyle = '--')
    ax.add_artist(target)
    
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
    savefig : bool, optional
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

    #annotate image
    target = plt.Circle((trialclass.target), 2.5, color='b')
    ax.add_artist(target)
    
    #testing 2target analysis
    line = np.polyfit(np.array([trialclass.target[0], trialclass.target_reverse[0]]), np.array([trialclass.target[1], trialclass.target_reverse[1]]), 1)

    x = np.arange(-60, 60)
    ax.plot(x, line[0]*x+line[1], ls='-', color = 'k') #line
    
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
    
#%%


def draw_hole_checks(data, idx_end, ax):
    k_times = data.k_hole_checks[data.k_hole_checks[:,1]<= idx_end] #crop at target
    
    colors_time_course = plt.get_cmap('cool') # plt.get_cmap('cool') #jet_r
    t_seq_hole = data.time[k_times[:,1]]/data.time[idx_end-1]
    # t_seq_traj = data.time/data.time[data.k_reward-1]
        
    #plots hole checks
    ax.scatter(data.r_arena_holes[k_times[:,0]][:,0], data.r_arena_holes[k_times[:,0]][:,1], 
               s=50, marker = 'o', facecolors='none', edgecolors=colors_time_course(t_seq_hole), 
               linewidths=2.)
    return ax

def draw_entrance(data, ax):
    #draw entrance
    for i, _ in enumerate(data.r_nose):
        if np.isnan(data.r_nose[i][0]): continue
        else:
            first_coord = data.r_nose[i]
            break
    entrance = plt.Rectangle((first_coord-3.5), 7, 7, fill=False, color='k', alpha=0.8, lw=3)
    ax.add_artist(entrance)
    return ax

def draw_heading(data, ax):
    coords = data.r_center
    coords_filter = coords[0::150]
    heading_filter = data.heading[0::150]
    for i, coord in enumerate(coords_filter):
        
        plt.axline(coord, slope=np.tan(np.radians(heading_filter[i])), color='red', label='axline')
    return ax
    

def plot_hole_checks(data, crop_at_target = True, time_limit = 'all', savefig=False):
    fig, ax = plt.subplots()
    
    draw_arena(data, ax)
    
    if crop_at_target: idx_end = data.k_reward
    else: idx_end = get_coords_timeLimit(data, time_limit)
    
    draw_hole_checks(data, idx_end, ax)
    
    plt.plot(data.r_nose[:idx_end,0], data.r_nose[:idx_end,1], color='k', alpha=0.5) #plot path
    # ax.scatter(data.r_nose[:idx_target,0], data.r_nose[:idx_target,1], s=1.5, facecolors=colors_time_course(t_seq_traj[:idx_target])) #plot path with colours
    
    
    # draw_heading(data, ax)
    
    # draw target
    target = plt.Circle((data.target), 2.5 , color='b', alpha=1)
    ax.add_artist(target)
    
    draw_entrance(data, ax)
    
    if savefig == True:
        plt.savefig(ROOT_DIR+f'/figures/HoleChecks_{data.exp}_M{data.mouse_number}_T{data.trial}.png', dpi=600, bbox_inches='tight', pad_inches = 0)
    
    plt.show()


if __name__ == '__main__': #only runs this function if the script top level AKA is running by itself
    data = plib.TrialData()
    data.Load('2024-02-15', '105', '14')
    print('Mouse %s Trial %s'%(data.mouse_number, data.trial))
    plot_hole_checks(data, crop_at_target=False, time_limit = '5min')


    pass