# -*- coding: utf-8 -*-
"""
Created on Wed May 05 15:15:48 2021
Calculate vector difference between mouse direction and food direction
@author: Kelly
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('relative_target\mouse_13\mpos_07Oct2019_trial_4_startloc_SW_day_7.mat')

# %%
# find index when mouse is in a particular x y location
def find_target_idx(x, y, coords):
    coords = np.array(coords, dtype=np.float64)
    radius = 5  # change this radius if you want the target to be bigger or smaller, normally 5
    time_in_range = ((coords >= [x - radius, y - radius]) & (coords <= [x + radius, y + radius])).all(axis=1)
    timepoint = np.around(np.where(abs(time_in_range) == True)[0])
    if timepoint.size == 0:
        timepoint = 0.  # turns nan value to zero
        print("WARNING target never reached")
    else: timepoint = timepoint[0]
    return timepoint


# automatically detect end point
def get_coords_auto():
    if find_target_idx(mat['r_target'].flatten()[0], mat['r_target'].flatten()[1], mat['r_nose']) != 0.: #checks if target was found
        idx_end = int(find_target_idx(mat['r_target'].flatten()[0], mat['r_target'].flatten()[1], mat['r_nose']))
    else: #if mouse never reaches target, plot the whole trajectory
        idx_end = int(len(mat['r_nose']))

    return idx_end


idx_end = get_coords_auto()


np.subtract(mat['r_target'],mat['r_center'])

# calculate difference in angle between two vectors in degrees
def calc_angle_diff(a, b):
    theta = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) * (180 / np.pi)
    return theta


#%%
# calculate angle from mouse towards target
target_vector = np.array([mat['r_target'].flatten()[0] - mat['r_center'][:idx_end, 0],
                          mat['r_target'].flatten()[1] - mat['r_center'][:idx_end, 1]])
target_direction = np.arctan(target_vector[0] / target_vector[1]) * (180 / np.pi)

# calculate mouse direction from nose point and center point until it reaches target
# should probably change this to calculate direction from movement vector
mouse_vector = np.array([mat['r_nose'][:idx_end, 0] - mat['r_center'][:idx_end, 0],
                         mat['r_nose'][:idx_end, 1] - mat['r_center'][:idx_end, 1]])
mouse_direction = np.arctan(mouse_vector[0] / mouse_vector[1]) * (180 / np.pi)

# calculate difference in angle between mouse direction and target direction
angle_difference = np.array([])
i = 0
while i <= idx_end - 1:
    angle_difference = np.append(angle_difference, calc_angle_diff(target_vector[:, i], mouse_vector[:, i]))
    i = i + 1

#%%

fig, ax = plt.subplots(nrows=2, figsize=(8, 6))
ax[0].plot(mat['time'][:, :idx_end].flatten(), mouse_direction)  # plot direction vs time
ax[1].plot(mat['time'][:, :idx_end].flatten(), angle_difference)  # plot difference in direction vs time
ax[0].set_title('Mouse: %s, Trial %s' % (mat['mouse_number'][0], mat['trial'][0]), size=14)
ax[1].set_title('Angle difference vs time')
ax[0].set_ylabel('degree')
ax[1].set_ylabel('degree')
plt.xlabel('time(s)')
plt.show()

# plot coords
# fig, ax = plt.subplots()
# ax.plot(mat['r_nose'][:idx_end,0].flatten(), mat['r_nose'][:idx_end,1].flatten(), ls='-', color = 'red')
# plt.show()