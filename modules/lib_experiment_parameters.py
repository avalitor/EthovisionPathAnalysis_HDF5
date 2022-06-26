# -*- coding: utf-8 -*-
"""
Created on Thu May 12 20:45:02 2022

contains experiment parameters
pilot experiment data currently missing

Input Variables:
    
    experiment: str
        the experiment date, written as YYYY-MM-DD
        
    trial_condition: str
        trial written in excel file, such as 13 or Probe
        
    entrance: str
        two letter direction of the entrance, written in the excel file

@author: Kelly
"""


def check_reverse(experiment, trial_condition): #check if trial is a reversal
    
    if trial_condition.startswith(u'R') == True or trial_condition.startswith(u'Flip') == True: is_reverse = True
    else: is_reverse = False
    if experiment == '2021-06-22' or experiment == '2021-11-19':
        if trial_condition.startswith(u'Probe') or trial_condition.startswith(u'R'): is_reverse = True
        elif trial_condition.isdigit():
            if int(trial_condition) > 18: is_reverse = True
        else: is_reverse = False
    return is_reverse

#fetches background image
def set_background_image(experiment, is_reverse):
    if experiment == '2019-09-06' or experiment == '2019-10-07': background = 'BKGDimage-pilot.png'
    if experiment == '2019-12-11': background = 'BKGDimage-localCues.png'
    if experiment == '2021-03-08': background = 'BKGDimage-localCues_clear.png'
    if experiment == '2021-05-06': background = 'BKGDimage-localCues_Letter.png'
    if experiment == '2021-05-26': background = 'BKGDimage-localCues_LetterTex.png'
    if experiment == '2021-06-22' or experiment == '2021-11-19': background = 'BKGDimage-arenaB.png'
    if experiment == '2021-07-16': background = 'BDGDimage_arenaB_visualCues.png'
    if experiment == '2021-11-15': background = 'BKGDimage-Nov15.png'
    if experiment == '2021-07-30': 
        if is_reverse: background = 'BKGDimage_asymmREv.png'
        else: background = 'BKGDimage_asymm.png'
    if experiment == '2021-08-11': 
        if is_reverse: background = 'BKGDimage_3LocalCues_Reverse.png'
        else: background = 'BKGDimage_3LocalCues.png'
    if experiment == '2021-09-23': 
        if is_reverse: background = 'BKGDImage_ReverseCloseCues.png'
        else: background = 'BKGDImage_CloseCues.png'
    if experiment == '2022-02-13': background = 'BKGDimage-Luminosity.png'
    if experiment == '2022-03-21': 
        if is_reverse: background = 'BKGDimage_asymmRev.png'
        else: background = 'BKGDimage_asymm2.png'
    return background

#manually sets food target coordinates based on experiment
def set_target(experiment, entrance, trial_condition):
    if experiment == '2019-05-23' or experiment == 'pilot':
        target_coords = 20.47, -39.91
    elif experiment == '2019-09-06' or experiment == '2019-10-07' or experiment == '2019-06-18' or experiment == '2019-07-08':
        if trial_condition.isdigit() is True or trial_condition.startswith(u'Probe') or trial_condition.startswith(u'Dark'):
            if entrance == u'SW':
                target_coords = 11.07, -30.48
            if entrance == u'SE':
                target_coords = 35.64, 2.58
            if entrance == u'NE':
                target_coords = 2.88, 27.45
            if entrance == u'NW':
                target_coords = -21.68, -5.61
        elif trial_condition.startswith(u'R90'):
            if entrance == u'SW':
                target_coords = -21.68, -5.61
            if entrance == u'SE':
                target_coords = 11.07, -30.48
            if entrance == u'NE':
                target_coords = 35.64, 2.58
            if entrance == u'NW':
                target_coords = 2.88, 27.45
        elif trial_condition.startswith(u'R180'):
            if entrance == u'SW':
                target_coords = 2.88, 27.45
            if entrance == u'SE':
                target_coords = -21.68, -5.61
            if entrance == u'NE':
                target_coords = 11.07, -30.48
            if entrance == u'NW':
                target_coords = 35.64, 2.58
        elif trial_condition.startswith(u'R270'):
            if entrance == u'SW':
                target_coords = 35.64, 2.58
            if entrance == u'SE':
                target_coords = 2.88, 27.45
            if entrance == u'NE':
                target_coords = -21.68, -5.61
            if entrance == u'NW':
                target_coords = 11.07, -30.48
        elif trial_condition.startswith(u'Explore') or trial_condition.startswith(u'Training'): target_coords = float("NaN")
    elif experiment == '2019-12-11': target_coords = -4.32, 27.40
    elif experiment == '2021-03-08' or experiment == '2021-05-06' or experiment == '2021-05-26': target_coords = 24.47, 21.80
    elif experiment == '2021-06-22':
        if trial_condition.isdigit() is True and int(trial_condition) <= 18:
            if entrance == u'SW':
                target_coords = -38.64,-11.62
            if entrance == u'SE':
                target_coords = 12.25, -37.07
            if entrance == u'NE':
                target_coords = 37.54, 13.19
            if entrance == u'NW':
                target_coords = -13.19, 39.27
        else:
            if entrance == u'SW':
                target_coords = 22.15, 18.85
            if entrance == u'SE':
                target_coords = -19.01, 23.40
            if entrance == u'NE':
                target_coords = -22.93, -17.59
            if entrance == u'NW':
                target_coords = 17.91, -21.36
    elif experiment == '2021-07-16': target_coords = -29.21, -3.93
    elif experiment == '2021-07-30': target_coords = -2.2, 30.63
    elif experiment == '2021-08-11': target_coords = 28.12, 4.71
    elif experiment == '2021-09-23': target_coords = 18.58, -22.68
    elif experiment == '2021-11-15': target_coords = 29.13, 2.68
    elif experiment == '2021-11-19':
        if trial_condition.isdigit() is True and int(trial_condition) <= 18:
            if entrance == u'SW':
                target_coords = -38.32, -13.51
            if entrance == u'SE':
                target_coords = 12.88, -37.54
            if entrance == u'NE':
                target_coords = 37.54, 13.51
            if entrance == u'NW':
                target_coords = -13.51, 38.80
        else:
            if entrance == u'SW':
                target_coords = 21.99, 18.85
            if entrance == u'SE':
                target_coords = -19.32, 22.77
            if entrance == u'NE':
                target_coords = -22.62, -18.53
            if entrance == u'NW':
                target_coords = 18.38, -21.83
    elif experiment == '2022-02-13': target_coords = 28.82, 2.99
    elif experiment == '2022-03-21': target_coords = 28.71, 3.45
    return target_coords

#sets the rotationally equivalent location of the target, only use during rotation trials
def set_reverse_target(experiment, entrance, trial_condition):
    if experiment == '2019-05-23':
        reverse_target_coords = -6.32, 36.62
    elif experiment == '2019-09-06' or experiment == '2019-10-07' or experiment == '2019-07-08':
        if trial_condition.isdigit() is True:
            if entrance == u'SW':
                reverse_target_coords = 11.07, -30.48
            if entrance == u'SE':
                reverse_target_coords = 35.64, 2.58
            if entrance == u'NE':
                reverse_target_coords = 2.88, 27.45
            if entrance == u'NW':
                reverse_target_coords = -21.68, -5.61
        elif trial_condition.startswith(u'R90'):
            if entrance == u'SW':
                reverse_target_coords = 35.64, 2.58
            if entrance == u'SE':
                reverse_target_coords = 2.88, 27.45
            if entrance == u'NE':
                reverse_target_coords = -21.68, -5.61
            if entrance == u'NW':
                reverse_target_coords = 11.07, -30.48
        elif trial_condition.startswith(u'R180'):
            if entrance == u'SW':
                reverse_target_coords = 11.07, -30.48
            if entrance == u'SE':
                reverse_target_coords = 35.64, 2.58
            if entrance == u'NE':
                reverse_target_coords = 2.88, 27.45
            if entrance == u'NW':
                reverse_target_coords = -21.68, -5.61
        elif trial_condition.startswith(u'R270'):
            if entrance == u'SW':
                reverse_target_coords = 11.07, -30.48
            if entrance == u'SE':
                reverse_target_coords = 35.64, 2.58
            if entrance == u'NE':
                reverse_target_coords = 2.88, 27.45
            if entrance == u'NW':
                reverse_target_coords = -21.68, -5.61
    elif experiment == '2019-12-11': reverse_target_coords = 38.89, -0.51
    elif experiment == '2021-03-08' or experiment == '2021-05-06' or experiment == '2021-05-26': reverse_target_coords = -19.61, -16.63
    if experiment == '2021-06-22':
        if entrance == u'SW':
            reverse_target_coords = -38.64, -11.62
        if entrance == u'SE':
            reverse_target_coords = 12.25, -37.07
        if entrance == u'NE':
            reverse_target_coords = 37.54, 13.19
        if entrance == u'NW':
            reverse_target_coords = -13.19, 39.27
    elif experiment == '2021-07-16': reverse_target_coords = 27.33, 4.4
    elif experiment == '2021-07-30': reverse_target_coords = 6.75, -26.23
    elif experiment == '2021-08-11': reverse_target_coords = -29.21, -3.77
    elif experiment == '2021-09-23': reverse_target_coords = -18.90, 22.52
    elif experiment == '2021-11-15': reverse_target_coords = -28.19, -2.68
    elif experiment == '2021-11-19':
        if entrance == u'SW':
            reverse_target_coords = -38.32, -13.51
        if entrance == u'SE':
            reverse_target_coords = 12.88, -37.54
        if entrance == u'NE':
            reverse_target_coords = 37.54, 13.51
        if entrance == u'NW':
            reverse_target_coords = -13.51, 38.80
    elif experiment == '2022-02-13': reverse_target_coords = -28.35, -2.36
    elif experiment == '2022-03-21': reverse_target_coords = -28.08, -3.29
    return reverse_target_coords