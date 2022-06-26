# -*- coding: utf-8 -*-
"""
Created on Sun May 15 14:29:42 2022

plots mouse trajectory
requires lib_plot_mouse_trajectory

@author: Kelly
"""

from modules import lib_process_data_to_mat as plib
from modules import lib_plot_mouse_trajectory as pltlib


''' plots a single mouse trajectory '''
objs = plib.TrialData()
objs.Load('2019-12-11', '18', 1)
print('Mouse %s Trial %s'%(objs.mouse_number, objs.trial))

pltlib.plot_single_traj(objs, cropcoords = True)


'''plots multiple trajectories'''
# objs = [plib.TrialData() for i in range(2)]
# objs[0].Load('2019-12-11', 19, '9')
# objs[1].Load('2019-12-11', 20, '10')

# print(objs[0].protocol_name)

# pltlib.plot_multi_traj(objs, savefig = False)

'''plots heatmap of two experiments'''
# objs = [plib.TrialData() for i in range(8)]
# objs[0].Load('2021-07-16', 37, 'Probe')
# objs[1].Load('2021-07-16', 38, 'Probe')
# objs[2].Load('2021-07-16', 39, 'Probe')
# objs[3].Load('2021-07-16', 40, 'Probe')
# objs[4].Load('2021-11-15', 53, 'Probe')
# objs[5].Load('2021-11-15', 54, 'Probe')
# objs[6].Load('2021-11-15', 55, 'Probe')
# objs[7].Load('2021-11-15', 56, 'Probe')

# objs = [plib.TrialData() for i in range(8)]
# objs[0].Load('2019-09-06', 9, 'Probe')
# objs[1].Load('2019-09-06', 10, 'Probe')
# objs[2].Load('2019-09-06', 11, 'Probe')
# objs[3].Load('2019-09-06', 12, 'Probe')
# objs[4].Load('2019-10-07', 13, 'Probe')
# objs[5].Load('2019-10-07', 14, 'Probe')
# objs[6].Load('2019-10-07', 15, 'Probe')
# objs[7].Load('2019-10-07', 16, 'Probe')


# pltlib.plot_heatmap(objs, '2min', False)