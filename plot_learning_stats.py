# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:44:32 2022

Makes percent bars for static and changing entrances
for spatial learning manuscript

@author: Kelly
"""

import modules.lib_plot_learning_stats as ls
import modules.calc_latency_distance_speed as calc
from modules import lib_process_data_to_mat as plib

'''
PLOT PERCENT BAR
*****************
    
Rotating Entrances
'''
# d = calc.calc_search_bias(['2021-07-16', '2021-11-15'], 'Probe', '2min')
# ls.plot_percent_bar(d)

'''
Static Entrances
'''
# d = calc.calc_search_bias(['2019-09-06', '2019-10-07'], 'Probe', '2min')
# ls.plot_percent_bar(d)

'''3 Local Cues'''
# d = calc.calc_search_bias(['2019-12-11','2021-08-11'], 'Probe', '2min')
# ls.plot_percent_bar(d)

# d = calc.calc_search_bias(['2023-07-07'], 'Probe', '2min', 15.)
# ls.plot_percent_bar(d)

'''individual trial'''
# exp = plib.TrialData()
# exp.Load('2023-08-15', '91', 'Reverse')
# d = calc.compare_target_dwell(exp, exp.target_reverse, time_limit = '5min', radius = 15.)
# ls.plot_percent_bar(d)

'''
LATENCY, DISTANCE, SPEED LEARNING CURVES
***************************************

Single trial REL data
'''
# exp = plib.TrialData()
# exp.Load('2019-09-06', '12', 'R180 1')
# print(f'Mouse {exp.mouse_number} Trial {exp.trial}')
# latency, distance, speed = calc.calc_lat_dist_sped(exp, custom_target=exp.target_reverse)

'''
Whole experiemnt before probe
'''
rotate = calc.iterate_all_trials(['2022-08-12'], continuous= False, training_trials_only=True)
ls.plot_latency(rotate, log=True, savefig = False)
ls.plot_distance(rotate, log=True, savefig = False)
ls.plot_speed(rotate, savefig = False)
calc.curve_pValue(rotate)

'''
3 Local Cues
'''
# loc = calc.iterate_all_trials(['2019-12-11','2021-08-11'], continuous= False)
# ls.plot_latency(loc, log=True, savefig = False)
# ls.plot_distance(loc, log=True, savefig = False)
# ls.plot_speed(loc, savefig = False)
# calc.curve_pValue(loc)

'''
2 Target REL
'''
# ttg = calc.iterate_all_trials(['2022-08-12'], continuous= False)
# ls.plot_latency(ttg, bestfit=False, log=True, savefig = False)
# ls.plot_distance(ttg, bestfit=False, log=True, savefig = False)
# ls.plot_speed(ttg, bestfit=False, savefig = False)
# calc.curve_pValue(ttg)

'''Compare 2 learning curves'''
# dark_trial = calc.iterate_all_trials(['2023-01-11','2022-08-12'], training_trials_only = True, continuous= False)
# dark, light = dark_trial['Distance'][['77','78','79','80']], dark_trial['Distance'][['69','70','71','72']]
# ls.plot_compare_curves(dark, light, 'Trained in Dark', 'Trained in Light', show_sig = True, log = False)

# sex_trial = calc.iterate_all_trials(['2022-08-12','2022-09-20'], training_trials_only = True, continuous= False)
# male, female = sex_trial['Speed'][['69','70','71','72']], sex_trial['Speed'][['73','74','75','76']]
# ls.plot_compare_curves(male, female, 'Male', 'Female', show_sig = True, log = False)

# ATRX_trial = calc.iterate_all_trials(['2023-07-07', '2023-08-15'], training_trials_only = False, continuous= False)
# KO, WT = ATRX_trial['Distance'][['85','88','91','92','93','94','98']], ATRX_trial['Distance'][['86','87','89','90','95','96','97']]
# ls.plot_compare_curves(KO, WT, 'KO', 'WT', "Distance (cm)", show_sig = True, log = True, crop_trial = False, savefig=False)

# compare_trial = calc.iterate_all_trials(['2023-10-16', '2023-12-18', '2024-02-12', '2024-02-15'], training_trials_only = False, continuous= False)
# old, new = compare_trial['Distance'][['101', '102', '103']], compare_trial['Distance'][['104', '105']]
# ls.plot_compare_curves(old, new, 'No 30min', 'Yes 30min', "Distance (cm)", show_sig = True, log = True, crop_trial = False, savefig=False)

'''
DISTANCE BETWEEN TARGETS
************************
'''
# exp = plib.TrialData()
# exp.Load('2023-02-13', 84, 'Probe3')
# max_spread =  calc.calc_traj_spread(exp) #calculates max distance from optimal line

# dist_AB = calc.calc_dist_bw_points(exp.r_center, exp.target_reverse, exp.target) #calculate distance between two points
# print(f'Distance between targets for Mouse {exp.mouse_number} during {exp.trial} is {round(dist_AB, 2)} cm')