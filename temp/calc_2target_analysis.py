# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:21:36 2022

Analysis for Ciara's neurogenesis knockout experiment
2 Target REL

@author: Kelly
"""

import modules.lib_plot_learning_stats as ls
import modules.calc_latency_distance_speed as calc
from modules import lib_process_data_to_mat as plib
from modules import lib_plot_mouse_trajectory as pltlib

#%%
'''plot single traj'''
exp = plib.TrialData()
exp.Load('2022-10-11', '1', '19')
print('Mouse %s Trial %s'%(exp.mouse_number, exp.trial))

pltlib.plot_single_traj(exp, cropcoords = True, savefig=False)

#%%
'''
LATENCY, DISTANCE, SPEED LEARNING CURVES
***************************************
'''
ttg = calc.iterate_all_trials(['2022-10-11'], continuous= False)
ls.plot_latency(ttg, bestfit=False, log=True, savefig = False)
ls.plot_distance(ttg, bestfit=False, log=True, savefig = False)
ls.plot_speed(ttg, bestfit=False, savefig = False)
calc.curve_pValue(ttg)

#get single trial stats
lat, dist, speed = calc.calc_lat_dist_sped(exp)
print(f'Latency is {round(lat, 2)} s, Distance is {round(dist, 2)} cm, Speed is {round(speed, 2)} cm/s')
#%%
'''
PROBE1 VS PROBE2 DISTANCE
***************************
'''
test = 8

'''Plot Probe 1, between targets'''
exp = plib.TrialData()
exp.Load('2022-10-11', test, 'Probe') #load the experiment trial
print(f'Mouse {exp.mouse_number} Trial {exp.trial}')

dist_AB = calc.calc_dist_bw_points(exp.r_center, exp.target_reverse, exp.target) #calculate distance between two points
print(f'Distance between targets for Mouse {exp.mouse_number} during {exp.trial} is {round(dist_AB, 2)} cm')

pltlib.plot_single_traj(exp, crop_interval=(exp.target_reverse, exp.target), savefig=False)

'''Plot Probe 2, between targets'''
exp = plib.TrialData()
exp.Load('2022-10-11', test, 'Probe 2')
print('Mouse %s Trial %s'%(exp.mouse_number, exp.trial))

dist_AB = calc.calc_dist_bw_points(exp.r_center, exp.target, exp.target_reverse)
print(f'Distance between targets for Mouse {exp.mouse_number} during {exp.trial} is {round(dist_AB, 2)} cm')

pltlib.plot_single_traj(exp, crop_interval=(exp.target, exp.target_reverse))


'''
PROBE2 DISTANCE: BETWEEN TARGETS VS RANDOM
******************************************
'''
exps = [plib.TrialData() for i in range(2)]
exps[0].Load('2022-10-11', test-1, 'Probe 2') #load the experiment trial for rotated 90 coords
exps[1].Load('2022-10-11', test, 'Probe 2') #load the experiment trial
points_random = [exps[0].target, exps[0].target_reverse]
dist_rand = calc.calc_dist_bw_points(exps[1].r_center, points_random[1], points_random[0]) #calculate distance between two points
print(f'Distance between two arbitary points for Mouse {exps[1].mouse_number} during {exps[1].trial} is {round(dist_rand, 2)} cm')

pltlib.plot_single_traj(exps[1], crop_interval=(points_random[1], points_random[0]))
