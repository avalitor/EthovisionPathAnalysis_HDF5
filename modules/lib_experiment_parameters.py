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
        
    mouse: str
        mouse number

@author: Kelly
"""


def check_reverse(experiment, trial_condition): #check if trial is a reversal
    
    if trial_condition.startswith(u'R') == True or trial_condition.startswith(u'Flip') == True: is_reverse = True
    else: is_reverse = False
    if experiment == '2021-06-22' or experiment == '2021-11-19' or experiment == '2022-08-12' or experiment == '2022-09-20' or experiment == '2022-10-11':
        if trial_condition.startswith(u'Probe') or trial_condition.startswith(u'R'): is_reverse = True
        elif trial_condition.isdigit():
            if int(trial_condition) > 18: is_reverse = True
        else: is_reverse = False
    if experiment == '2023-02-13':
        if trial_condition.startswith(u'Probe'): is_reverse = True
        elif trial_condition.isdigit() is True and int(trial_condition) <= 12: is_reverse = False
        else: is_reverse = True
    return is_reverse

#fetches background image
def set_background_image(experiment, is_reverse, entrance, trial_condition):
    if experiment == '2019-05-23' or experiment == '2019-09-06' or experiment == '2019-10-07' or experiment == '2019-06-18': background = 'BKGDimage-pilot.png'
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
    if experiment == '2022-08-12': background = 'BKGDimage-220812.png'
    if experiment == '2022-09-20': 
        if trial_condition == 'Probe2': background = 'BKGDimage-20220920-probe2.png'
        else: background = 'BKGDimage-20220920.png'
    if experiment == '2022-10-11': background = 'BKGDimage-20221011.png'
    if experiment == '2022-11-04': background = 'BKGDimage-20221104.png'
    if experiment == '2023-01-11': background = 'BKGDimage-20230111.png'
    if experiment == '2023-02-13':
        if trial_condition.isdigit() is True or trial_condition == 'Probe' or trial_condition == 'Probe2':
            if entrance == u'SW':
                background = 'BKGDimage-barrier1.png'
            if entrance == u'SE':
                background = 'BKGDimage-barrier2.png'
            if entrance == u'NE':
                background = 'BKGDimage-barrier3.png'
            if entrance == u'NW':
                background = 'BKGDimage-barrier4.png'
        else: background = 'BKGDimage-20230111.png'
    if experiment == '2023-05-01': background = 'BKGDimage-20230501.png'
    if experiment == '2023-07-07': background = 'BKGDimage-20230707.png'
    if experiment == '2023-08-15': background = 'BKGDimage-20230815.png'
    if experiment == '2023-09-18': background = 'BKGDimage-20230918.png'
    if experiment == '2023-10-16': background = 'BKGDimage-20231016.png'
    if experiment == '2023-12-18': background = 'BKGDimage-20231218.png'
    if experiment == '2024-02-15' or experiment == '2024-02-12': background = 'BKGDimage-20240215.png'
    if experiment == '2024-02-06': background = 'BKGDimage-20240206.png'
    if experiment == '2024-05-06': background = 'BKGDimage-20240506.png'
    if experiment == '2024-09-23': background = 'BKGDimage-20240923.png'
    if experiment == '2024-06-27': background = 'BKGDimage-20240627.png'
    if experiment == '2024-11-08': background = 'BKGDimage-20241108.png'
    if experiment == '2024-11-28': background = 'BKGDimage-20241128.png'
    if experiment == '2025-05-07': background = 'BKGDimage-20250507.png'
    return background

def get_mouse_sex(experiment, mouse):
    if experiment == '2022-10-11':
        if mouse == '2' or mouse == '3' or mouse == '4' or mouse == '5' or mouse == '7': 
            mouse_sex = 'male'
        if mouse == '1' or mouse == '6' or mouse == '8':
            mouse_sex = 'female'
    if experiment == '2023-12-18':
        if mouse == '102': 
            mouse_sex = 'female'
        if mouse == '103':
            mouse_sex = 'male'
    if experiment == '2024-02-06':
        if mouse == '17': 
            mouse_sex = 'female'
        else:
            mouse_sex = 'male'
    if experiment == '2024-06-27':
        if mouse == '1': 
            mouse_sex = 'male'
        if mouse == '2':
            mouse_sex = 'female'
    return mouse_sex

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
    elif experiment == '2022-08-12':
        if (trial_condition.isdigit() is True and int(trial_condition) <= 18) or trial_condition == 'Probe': #target A
            if entrance == u'SW':
                target_coords = 39.68, 11.97
            if entrance == u'SE':
                target_coords = -11.18, 38.58
            if entrance == u'NE':
                target_coords = -37.48, -11.65
            if entrance == u'NW':
                target_coords = 12.28, -38.74
        else: #target B
            if entrance == u'SW':
                target_coords = -22.36, -17.80
            if entrance == u'SE':
                target_coords = 18.27, -22.99
            if entrance == u'NE':
                target_coords = 23.46, 18.27
            if entrance == u'NW':
                target_coords = -17.32, 22.83
    elif experiment == '2022-09-20':
        if trial_condition.isdigit() is True and int(trial_condition) <= 18:
            if entrance == u'SW':
                target_coords = 34.06, 12.14
            if entrance == u'SE':
                target_coords = -16.24, 38.48
            if entrance == u'NE':
                target_coords = -42.89, -11.98
            if entrance == u'NW':
                target_coords = 7.10, -39.11
        elif trial_condition == 'Probe2': #special aspect ratio
            if entrance == u'SW':
                target_coords = -22.80, -18.40
            if entrance == u'SE':
                target_coords = 17.93, -22.96
            if entrance == u'NE':
                target_coords = 22.49, 17.93
            if entrance == u'NW':
                target_coords = -17.77, 22.49           
        else: #target B
            if entrance == u'SW':
                target_coords = -27.75, -17.98
            if entrance == u'SE':
                target_coords = 13.40, -23.50
            if entrance == u'NE':
                target_coords = 18.13, 17.98
            if entrance == u'NW':
                target_coords = -22.71, 22.71
    elif experiment == '2022-10-11':
        if (trial_condition.isdigit() is True and int(trial_condition) <= 18) or trial_condition == 'Probe': #target A
            if entrance == u'SW':
                target_coords = -39.10, -11.94
            if entrance == u'SE':
                target_coords = 11.42,-37.68 
            if entrance == u'NE':
                target_coords = 38.39,12.37 
            if entrance == u'NW':
                target_coords = -12.51,39.53  
        else: #target B
            if entrance == u'SW':
                target_coords = 22.61,18.48
            if entrance == u'SE':
                target_coords = -18.63, 23.32
            if entrance == u'NE':
                target_coords = -23.60,-17.63
            if entrance == u'NW':
                target_coords = 17.63, -22.18
    elif experiment == '2022-11-04':
        if entrance == u'SW':
            target_coords = 38.90, 11.97
        if entrance == u'SE':
            target_coords = -11.65, 38.27
        if entrance == u'NE':
            target_coords = -38.11, -12.13
        if entrance == u'NW':
            target_coords = 11.65, -38.74 
    elif experiment == '2023-01-11':
        if trial_condition.startswith(u'R') == True:
            if entrance == u'SW':
                target_coords = 38.30, 12.06
            if entrance == u'SE':
                target_coords = -12.34, 39.29 
            if entrance == u'NE':
                target_coords = -39.01, -11.63
            if entrance == u'NW':
                target_coords = 11.49, -38.01
        else:
            if entrance == u'SW':
                target_coords = -39.01, -11.63
            if entrance == u'SE':
                target_coords = 11.49, -38.01
            if entrance == u'NE':
                target_coords = 38.30, 12.06
            if entrance == u'NW':
                target_coords = -12.34, 39.29 
    elif experiment == '2023-02-13':
        if (trial_condition.isdigit() is True and int(trial_condition) <= 12) or trial_condition == 'Probe': #target A
            if entrance == u'SW':
                target_coords = -39.01, -11.63
            if entrance == u'SE':
                target_coords = 11.63, -37.87
            if entrance == u'NE':
                target_coords = 38.01, 12.34
            if entrance == u'NW':
                target_coords = -12.48, 39.43  
        else: #target B
            if entrance == u'SW':
                target_coords = 22.98, 18.58
            if entrance == u'SE':
                target_coords = -18.58, 23.55
            if entrance == u'NE':
                target_coords = -23.26, -18.01
            if entrance == u'NW':
                target_coords = 17.73, -22.13
    elif experiment == '2023-05-01':
        if entrance == 'SW':
            target_coords = 38.64, 13.66
        if entrance == 'SE':
            target_coords = -13.35, 38.48
        if entrance == 'NE':
            target_coords = -37.54, -12.57
    elif experiment == '2023-07-07':
        if entrance == u'SW':
            target_coords = -38.44, -13.48
        if entrance == u'SE':
            target_coords = 13.05, -37.59
        if entrance == u'NE':
            target_coords = 38.01, 13.62
        if entrance == u'NW':
            target_coords = -13.48, 38.87
    elif experiment == '2023-08-15':
        if entrance == u'SW':
            target_coords = -39.00, -12.38
        if entrance == u'SE':
            target_coords = 11.67, -38.15
        if entrance == u'NE':
            target_coords = 38.43, 12.38
        if entrance == u'NW':
            target_coords = -12.53, 39.00
    elif experiment == '2023-09-18':
        if entrance == u'SW':
            target_coords = 38.90, 12.76
        if entrance == u'SE':
            target_coords = -11.97, 38.58
        if entrance == u'NE':
            target_coords = -37.48, -12.13
        if entrance == u'NW':
            target_coords = 12.60, -38.11
    elif experiment == '2023-10-16':
        target_coords = 38.10, 12.65
    elif experiment == '2023-12-18':
        if entrance == u'NE':
            target_coords = 38.10, 12.65
        if entrance == u'NW':
            target_coords = -12.65, 39.38
    elif experiment == '2024-02-12':
        target_coords = 1.85, -39.38
    elif experiment == '2024-02-15':
        target_coords = 39.38, 2.70
    elif experiment == '2024-02-06':
        if entrance == u'SW':
            target_coords = 38.59, 12.03
        if entrance == u'SE':
            target_coords = -11.09, 39.06
        if entrance == u'NE':
            target_coords = -37.96, -10.94
        if entrance == u'NW':
            target_coords = 11.56, -38.43
    elif experiment == '2024-05-06':
        if entrance == u'SW':
            target_coords = 37.91, 14.88
        if entrance == u'SE':
            target_coords = -14.41, 37.44
        if entrance == u'NE':
            target_coords = -37.75, -14.26
        if entrance == u'NW':
            target_coords = 14.26, -37.44
    elif experiment == '2024-09-23':
        if entrance == u'SW':
            target_coords = 39.21, 13.70
        if entrance == u'SE':
            target_coords = -12.28, 38.74
        if entrance == u'NE':
            target_coords = -37.64, -12.76
        if entrance == u'NW':
            target_coords = 13.86, -37.95
    elif experiment == '2024-06-27':
        targA_trials = [*range(1, 19, 1)]+[32, 34, 36, 37, 39, 41, 44, 46, 48]
        targB_trials = [*range(19, 31, 1)]+[31, 33, 35, 38, 40, 42, 43, 45, 47]
        if (trial_condition.isdigit() is True and int(trial_condition) in targA_trials) or trial_condition == 'Probe' or trial_condition.startswith(u'Habituation'):
            if entrance == u'NE': #target A
                target_coords = 39.67,2.27
            if entrance == u'NW':
                target_coords = -2.42, 40.52
        elif (trial_condition.isdigit() is True and int(trial_condition) in targB_trials) or trial_condition == 'Probe2':
            if entrance == u'NE': #target B
                target_coords = -23.18, -18.20
            if entrance == u'NW':
                target_coords = 18.06, -22.61
    elif experiment == '2024-11-08':
        if entrance == u'SW':
            target_coords = 38.05, 13.09
        if entrance == u'SE':
            target_coords = -3.23, 40.20
        if entrance == u'NE':
            target_coords = -38.35, -12.94
        if entrance == u'NW':
            target_coords = 13.09, -38.35
    elif experiment == '2024-11-28':
        if entrance == u'SW':
            target_coords = 38.40, 13.73
        if entrance == u'SE':
            target_coords = -2.93, 40.72
        if entrance == u'NE':
            target_coords = -38.09, -12.49
        if entrance == u'NW':
            target_coords = 12.96, -37.94
    elif experiment == '2025-05-07':
        if trial_condition.isdigit() is True and int(trial_condition) <= 45 or trial_condition == 'probe 1': #target A
            if entrance == u'SW':
                target_coords = 38.09, 13.11
            if entrance == u'SE':
                target_coords = -12.65, 38.40
            if entrance == u'NE':
                target_coords = -38.87, -12.65
            if entrance == u'NW':
                target_coords = 0,0
        else: #Target B
            if entrance == u'SW':
                target_coords = -19.12, -23.91
            if entrance == u'SE':
                target_coords = 23.44, -18.97
            if entrance == u'NE':
                target_coords = 18.66, 23.91
            if entrance == u'NW':
                target_coords = 0,0
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
    elif experiment == '2022-08-12': #normal target A
        if entrance == u'SW':
            reverse_target_coords = 39.68, 11.97
        if entrance == u'SE':
            reverse_target_coords = -11.18, 38.58
        if entrance == u'NE':
            reverse_target_coords = -37.48, -11.65
        if entrance == u'NW':
            reverse_target_coords = 12.28, -38.74
    elif experiment == '2022-09-20':
        if trial_condition == "Probe2": #normal target A
            if entrance == u'SW':
                reverse_target_coords = 38.85, 11.80
            if entrance == u'SE':
                reverse_target_coords = -11.80, 38.37
            if entrance == u'NE':
                reverse_target_coords = -38.06, -12.27
            if entrance == u'NW':
                reverse_target_coords = 11.80, -39.16
        else: #normal target A
            if entrance == u'SW':
                reverse_target_coords = 34.06, 12.14
            if entrance == u'SE':
                reverse_target_coords = -16.24, 38.48
            if entrance == u'NE':
                reverse_target_coords = -42.89, -11.98
            if entrance == u'NW':
                reverse_target_coords = 7.10, -39.11
    elif experiment == '2022-10-11': #normal target A
        if entrance == u'SW':
            reverse_target_coords = -39.10, -11.94
        if entrance == u'SE':
            reverse_target_coords = 11.42,-37.68 
        if entrance == u'NE':
            reverse_target_coords = 38.39,12.37 
        if entrance == u'NW':
            reverse_target_coords = -12.51,39.53
    elif experiment == '2022-11-04':
        if entrance == u'SW':
            reverse_target_coords = 11.65, -38.74
        if entrance == u'SE':
            reverse_target_coords = 38.90, 11.97
        if entrance == u'NE':
            reverse_target_coords = -11.65, 38.27
        if entrance == u'NW':
            reverse_target_coords = -38.11, -12.13
    elif experiment == '2023-01-11':
        if entrance == u'SW':
            reverse_target_coords = -39.01, -11.63 
        if entrance == u'SE':
            reverse_target_coords = 11.49, -38.01
        if entrance == u'NE':
            reverse_target_coords = 38.30, 12.06
        if entrance == u'NW':
            reverse_target_coords = -12.34, 39.29
    elif experiment == '2023-02-13':
        if trial_condition == "Probe":
            if entrance == u'SW':
                reverse_target_coords = 22.98, 18.58
            if entrance == u'SE':
                reverse_target_coords = -18.58, 23.55
            if entrance == u'NE':
                reverse_target_coords = -23.26, -18.01
            if entrance == u'NW':
                reverse_target_coords = 17.73, -22.13
        else:
            if entrance == u'SW':
                reverse_target_coords = -39.01, -11.63
            if entrance == u'SE':
                reverse_target_coords = 11.63, -37.87
            if entrance == u'NE':
                reverse_target_coords = 38.01, 12.34
            if entrance == u'NW':
                reverse_target_coords = -12.48, 39.43
    elif experiment == '2023-07-07':
        if entrance == u'SW':
            reverse_target_coords = 38.01, 13.62
        if entrance == u'SE':
            reverse_target_coords = -13.48, 38.87
        if entrance == u'NE':
            reverse_target_coords = -38.44, -13.48
        if entrance == u'NW':
            reverse_target_coords = 13.05, -37.59
    elif experiment == '2023-08-15':
        if entrance == u'SW':
            reverse_target_coords = 38.43, 12.38
        if entrance == u'SE':
            reverse_target_coords = -12.53, 39.00
        if entrance == u'NE':
            reverse_target_coords = -39.00, -12.38
        if entrance == u'NW':
            reverse_target_coords = 11.67, -38.15
    elif experiment == '2024-06-27':
        targA_trials = [32, 34, 36, 37, 39, 41, 44, 46, 48]
        targB_trials = [*range(19, 31, 1)]+[31, 33, 35, 38, 40, 42, 43, 45, 47]
        if (trial_condition.isdigit() is True and int(trial_condition) in targB_trials) or trial_condition == 'Probe2':
            if entrance == u'NE': #target A
                reverse_target_coords = 39.67,2.27
            if entrance == u'NW':
                reverse_target_coords = -2.42, 40.52
        elif trial_condition.isdigit() is True and int(trial_condition) in targA_trials:
            if entrance == u'NE': #target B
                reverse_target_coords = -23.18, -18.20
            if entrance == u'NW':
                reverse_target_coords = 18.06, -22.61
    return reverse_target_coords