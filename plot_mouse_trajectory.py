# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:29:42 2022

plots mouse trajectory
requires lib_plot_mouse_trajectory

@author: Kelly
"""

from modules import lib_process_data_to_mat as plib
from modules import lib_plot_mouse_trajectory as pltlib


'''plot single traj'''
# exp = plib.TrialData()
# exp.Load('2022-11-04', '4', '21')
# print('Mouse %s Trial %s'%(exp.mouse_number, exp.trial))

# pltlib.plot_single_traj(exp, cropcoords = True, savefig=False)

''' plots a single mouse trajectory, cut off at REL target '''
# objs = plib.TrialData()
# objs.Load('2022-10-11', 1, 'Probe 2')
# print('Mouse %s Trial %s'%(objs.mouse_number, objs.trial))

# pltlib.plot_single_traj(objs, crop_end_custom = objs.target_reverse)

'''plots single trajectory between two points'''
# exp = plib.TrialData()
# exp.Load('2022-10-11', '1', 'Probe')
# print('Mouse %s Trial %s'%(exp.mouse_number, exp.trial))

# pltlib.plot_single_traj(exp, crop_interval=(exp.target, exp.target_reverse))


'''plots multiple trajectories and REL'''
# objs = [plib.TrialData() for i in range(2)]
# objs[0].Load('2022-11-04', 4, '20')
# objs[1].Load('2022-11-04', 4, 'R90')

# print(objs[0].protocol_name)

# pltlib.plot_multi_traj(objs, align_entrance=False, crop_rev = False, savefig = False)

'''plots multiple trajectories with custom coordinates'''
# objs = [plib.TrialData() for i in range(2)]
# objs[0].Load('2019-10-07', 14, 'R180 6')
# objs[1].Load('2019-10-07', 14, 'R270 1')

# print(objs[0].protocol_name)

# pltlib.plot_multi_traj(objs, crop_end_custom=[objs[0].target, (2.88, 27.45)], savefig = True)


'''plot multiple trajectories on different graphs'''
# objs = [plib.TrialData() for i in range(8)]
# objs[0].Load('2022-10-11', 1, 'Probe 2')
# objs[1].Load('2022-10-11', 2, 'Probe 2')
# objs[2].Load('2022-10-11', 3, 'Probe 2')
# objs[3].Load('2022-10-11', 4, 'Probe 2')
# objs[4].Load('2022-10-11', 5, 'Probe 2')
# objs[5].Load('2022-10-11', 6, 'Probe 2')
# objs[6].Load('2022-10-11', 7, 'Probe 2')
# objs[7].Load('2022-10-11', 8, 'Probe 2')

# for i in objs:
#     pltlib.plot_single_traj(i, crop_end_custom = i.target_reverse, savefig=True)
    

'''plots heatmap of two experiments'''
# objs = [plib.TrialData() for i in range(8)] #random entrance
# objs[0].Load('2021-07-16', 37, 'Probe')
# objs[1].Load('2021-07-16', 38, 'Probe')
# objs[2].Load('2021-07-16', 39, 'Probe')
# objs[3].Load('2021-07-16', 40, 'Probe')
# objs[4].Load('2021-11-15', 53, 'Probe')
# objs[5].Load('2021-11-15', 54, 'Probe')
# objs[6].Load('2021-11-15', 55, 'Probe')
# objs[7].Load('2021-11-15', 56, 'Probe')

# objs = [plib.TrialData() for i in range(8)] #static
# objs[0].Load('2019-09-06', 9, 'Probe')
# objs[1].Load('2019-09-06', 10, 'Probe')
# objs[2].Load('2019-09-06', 11, 'Probe')
# objs[3].Load('2019-09-06', 12, 'Probe')
# objs[4].Load('2019-10-07', 13, 'Probe')
# objs[5].Load('2019-10-07', 14, 'Probe')
# objs[6].Load('2019-10-07', 15, 'Probe')
# objs[7].Load('2019-10-07', 16, 'Probe')

# objs = [plib.TrialData() for i in range(4)] #3 local cue, not rotating correctly
# objs[0].Load('2019-12-11', 17, 'Probe')
# objs[1].Load('2019-12-11', 18, 'Probe')
# objs[2].Load('2019-12-11', 19, 'Probe')
# objs[3].Load('2019-12-11', 20, 'Probe')
# objs[0].Load('2021-08-11', 45, 'Probe')
# objs[1].Load('2021-08-11', 46, 'Probe')
# objs[2].Load('2021-08-11', 47, 'Probe')
# objs[3].Load('2021-08-11', 48, 'Probe')

# pltlib.plot_heatmap(objs, '2min', False)


# objs = [plib.TrialData() for i in range(1)] #3 local cue, not rotating correctly
# objs[0].Load('2022-11-04', 1, 'Probe')
# objs[1].Load('2022-11-04', 3, 'Probe')

objs = plib.TrialData()
objs.Load('2022-11-04', 1, 'Probe')
pltlib.plot_heatmap(objs, '2min', False)