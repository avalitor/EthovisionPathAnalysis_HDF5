# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:42:23 2023

detects arena and hole coordinates based on opencv

@author: Kelly
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules import lib_process_data_to_mat as plib

def detect_arena_circle(image_path, mask_sensitivity=40, scaling = 500.):
    img = cv2.imread(image_path)
    scale_ratio = img.shape[0] / scaling #gets image scale ratio for height
    img = cv2.resize(img, (int(img.shape[1]/scale_ratio), int(img.shape[0]/scale_ratio)), interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convet to grayscale
    gray = cv2.bilateralFilter(gray, 11, 17, 17) #blur image
    
    #this is the magic detect circles function
    circle_arena = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT,dp=1,minDist=20,
                                param1=80,
                                param2=mask_sensitivity, #increase to reduce circles
                                minRadius=50,maxRadius=200)
    
    circle_arena = np.uint16(np.around(circle_arena)) #convert to int
    
    for i in circle_arena[0,:]:
        # draw the outer circle
        cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0), 2) #center, radius, colour, thickness
        cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle
        main_circle = [i[0],i[1],i[2]]
    plt.imshow(gray) #test to see if arena circle is detected. If done correctly, there should be a single circle
    return main_circle, gray

def detect_arena_hole_coords(main_circle, gray):
    mask = np.zeros(gray.shape[:2], dtype="uint8") #create dark image of same size
    cv2.circle(mask, (main_circle[0], main_circle[1]), main_circle[2], 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask) #apply the mask

    # threshold
    th, threshed = cv2.threshold(masked, 100, 255, 
              cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #this is the magic "find small circles" function
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    holes = []
    for c in cnts:
        if cv2.contourArea(c) <30: # checks if hole is smaller than a certan area
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            cv2.circle(masked,(cX,cY),4,(0,255,0), 2) #draw circle center, radius, colour, thickness
            holes.append((cX,cY))
            
    plt.imshow(masked)
    return holes

def transpose_coords(holes, arena, current_extent = (499., 888.), target_extent = [-151.26, 150.94, -84.66, 85.29]):
    '''transform coordinates of arena and holes to be compatible with ethovison trial coordinates'''
    
    current_center = current_extent[1]/2, current_extent[0]/2
    cnt_norm = np.asarray(holes) - np.asarray(current_center) #move coords to center
    arena_norm = np.asarray(arena[:2]) - np.asarray(current_center) #do the same with the arena
    
    target_height = (target_extent[3]-target_extent[2])
    scale = current_extent[0]/target_height
    cnt_scaled = cnt_norm / scale #scale coords
    arena_scaled = np.append(arena_norm, arena[2]) / scale
    
    cnt_scaled = cnt_scaled * (1.,-1.) #flip positivity of y axis
    arena_scaled = arena_scaled  * (1.,-1., 1)

    return cnt_scaled, arena_scaled

if __name__ == '__main__':
    experiment = '2022-08-12'
    exp = plib.TrialData()
    exp.Load(experiment, '*', 'Probe')
    arena_circle, gray = detect_arena_circle(os.path.join(cfg.ROOT_DIR, 'data', 'BackgroundImage', exp.bkgd_img), 
                                                 mask_sensitivity=60.)
    holes = detect_arena_hole_coords(arena_circle, gray)
    r_holes, arena_coords = transpose_coords(holes, arena_circle, gray.shape, exp.img_extent)