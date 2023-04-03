# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:23:11 2023

Analysis for Ben's experiments:
2022-08-12	2_target_REL		M69-72
2022-09-20	2_target_REL_female	M73-76
2023-01-11	dark_training		M77-80
2023-02-13	2_target_barrier	M81-84

@author: Kelly
"""

import numpy as np
import matplotlib.pyplot as plt
import modules.lib_plot_learning_stats as ls
import modules.calc_latency_distance_speed as calc
import modules.lib_process_data_to_mat as plib
import modules.lib_plot_mouse_trajectory as pltlib

#%%
'''plot single trajectory to target'''
exp = plib.TrialData()
exp.Load('2022-08-12', '69', 'Probe2')
print(f'Mouse {exp.mouse_number} Trial {exp.trial}')
pltlib.plot_single_traj(exp, crop_end_custom = exp.target_reverse, savefig=False)

#%%
'''plot single traj between targets'''
exp = plib.TrialData()
exp.Load('2022-09-20', 75, 'Probe2')
dist_AB = calc.calc_dist_bw_points(exp.r_center, exp.target_reverse, exp.target) #calculate distance between two points
print(f'Distance between targets for Mouse {exp.mouse_number} during {exp.trial} is {round(dist_AB, 2)} cm')
pltlib.plot_single_traj(exp, crop_interval=(exp.target_reverse, exp.target), savefig=False)

'''plot single traj between random targets'''
# pltlib.plot_single_traj(exp, crop_interval=(pltlib.rotate(exp.target_reverse, exp.arena_circle[:2], 270), pltlib.rotate(exp.target, exp.arena_circle[:2], 270)), savefig=False)

#%%
'''Plot multiple trajectories between targets
2-target male experiment'''
exps = [plib.TrialData() for i in range(4)]
exps[0].Load('2022-08-12', 69, 'Probe2')
exps[1].Load('2022-08-12', 70, 'Probe2')
exps[2].Load('2022-08-12', 71, 'Probe2')
exps[3].Load('2022-08-12', 72, 'Probe2')

# exps[0].Load('2022-09-20', 73, 'Probe2')
# exps[1].Load('2022-09-20', 74, 'Probe2')
# exps[2].Load('2022-09-20', 75, 'Probe2')
# exps[3].Load('2022-09-20', 76, 'Probe2')


# pltlib.plot_multi_traj(exps, crop_rev= True, savefig = False) #crop at reverse target

crop_intervals = [(exps[0].target, exps[0].target_reverse),
                  (exps[1].target, exps[1].target_reverse),
                  (exps[2].target, exps[2].target_reverse),
                  (exps[3].target, exps[3].target_reverse)]

pltlib.plot_multi_traj(exps, align_entrance=True, crop_interval=crop_intervals, savefig=False) #crop traj between targets

#%%
'''barrier experiment'''
exps = [plib.TrialData() for i in range(4)]
# exps[0].Load('2023-02-13', 81, 'Probe')
# exps[1].Load('2023-02-13', 82, 'Probe')
# exps[2].Load('2023-02-13', 83, 'Probe')
# exps[3].Load('2023-02-13', 84, 'Probe')

# exps[0].Load('2023-02-13', 81, 'Probe2')
# exps[1].Load('2023-02-13', 82, 'Probe2')
# exps[2].Load('2023-02-13', 83, 'Probe2')
# exps[3].Load('2023-02-13', 84, 'Probe2')

exps[0].Load('2023-02-13', 81, 'Probe3')
exps[1].Load('2023-02-13', 82, 'Probe3')
exps[2].Load('2023-02-13', 83, 'Probe3')
exps[3].Load('2023-02-13', 84, 'Probe3')

# exps[0].Load('2023-02-13', 81, 'Probe4')
# exps[1].Load('2023-02-13', 82, 'Probe4')
# exps[2].Load('2023-02-13', 83, 'Probe4')
# exps[3].Load('2023-02-13', 84, 'Probe4')

# pltlib.plot_multi_traj(exps, crop_rev= True, savefig = False) #crop at reverse target

crop_intervals = [(exps[0].target, exps[0].target_reverse),
                  (exps[1].target, exps[1].target_reverse),
                  (exps[2].target, exps[2].target_reverse),
                  (exps[3].target, exps[3].target_reverse)]
pltlib.plot_multi_traj(exps, align_entrance=True, crop_interval=crop_intervals, savefig=False) #crop traj between targets

#%%
'''
LATENCY, DISTANCE, SPEED LEARNING CURVES
***************************************
'''
ttg = calc.iterate_all_trials(['2023-01-11'], training_trials_only = True, continuous= False)
ls.plot_latency(ttg, bestfit=False, log=True, savefig = False)
ls.plot_distance(ttg, bestfit=False, log=True, savefig = False)
ls.plot_speed(ttg, bestfit=False, savefig = False)
p=calc.curve_pValue(ttg)

#get single trial stats
lat, dist, speed = calc.calc_lat_dist_sped(exp)
print(f'Latency is {round(lat, 2)} s, Distance is {round(dist, 2)} cm, Speed is {round(speed, 2)} cm/s')

#%%
'''
A-B Distance, Spread & Random Distance Comparison
'''
exp = [plib.TrialData() for i in range(12)]
exp[0].Load('2022-08-12', 69, 'Probe')
exp[1].Load('2022-08-12', 70, 'Probe')
exp[2].Load('2022-08-12', 71, 'Probe')
exp[3].Load('2022-08-12', 72, 'Probe')
exp[4].Load('2022-08-12', 69, 'Probe2')
exp[5].Load('2022-08-12', 70, 'Probe2')
exp[6].Load('2022-08-12', 71, 'Probe2')
exp[7].Load('2022-08-12', 72, 'Probe2')
exp[8].Load('2023-02-13', 81, 'Probe3')
exp[9].Load('2023-02-13', 82, 'Probe3')
exp[10].Load('2023-02-13', 83, 'Probe3')
exp[11].Load('2023-02-13', 84, 'Probe3')

numbers = calc.calc_spread_iterate_exp(exp)

spread =  np.array(calc.calc_traj_spread(exp[11])) #individual spreads

#%%
'''
Vector angle towards barrier points
'''
exp = plib.TrialData()
exp.Load('2023-02-13', 83, 'Probe3')
arena_center = exp.arena_circle[:2]
target_v, target_d, mouse_v, mouse_d = calc.vector_to_target(exp, coord_target = pltlib.rotate(exp.target, arena_center, 90), coord_start = None, coord_end = exp.target_reverse)

angle_difference = calc.iterate_angle_difference(target_v, mouse_v)

idx_end = pltlib.coords_to_target(exp.r_nose, exp.target_reverse)

fig, ax = plt.subplots(nrows=2, figsize=(8, 7))
ax[0].plot(exp.time[:idx_end], mouse_d)  # plot direction vs time
ax[1].plot(exp.time[:idx_end], angle_difference)  # plot difference in direction vs time
ax[0].set_title(f'Mouse: {exp.mouse_number}, Trial {exp.trial}', size=14)
ax[1].set_title('Angle difference vs. time')
ax[0].set_ylabel('degree')
ax[1].set_ylabel('degree')
plt.xlabel('time(s)')
plt.show()

