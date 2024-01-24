# -*- coding: utf-8 -*-
"""
Created on Thu May 12 19:21:54 2022

library to process excel data int hdf5 files
some functions are taken from code written by Mauricio Girardis

@author: Kelly
"""

import os
import numpy as np
import scipy.io
import pandas as pd
import glob #for file search

from modules import lib_experiment_parameters as params
from modules.config import DOCU, RAW_FILE_DIR, PROCESSED_FILE_DIR


'''
Helper Functions
'''

def try_or_default(f, default=np.nan, msg=''):
    #try to execute function f, if not print an error message. function must contain the error message
    try:
        return f()
    except:
        if len(msg)>0:
            print(msg)
        return default

def get_path_from_exp(exp, eth_file):
    #get excel file path from experiment and ethovision file variables. exp: str, eth_file: int
    try: path = glob.glob(glob.glob(RAW_FILE_DIR+'/'+exp+'*/')[0]+'*'+str(eth_file)+'.xlsx', 
                     recursive = True)[0] #finds file path based on ethovision trial number
    except: raise ValueError('The specified file does not exist ::: %s File %i'%(exp,eth_file))
    return path

'''
Main Data Storage Functions
'''

class TrialData(): #container to store all trial data and metadata
    
    def __init__(self, exp='', protocol_name='', protocol_description='',
                  eth_file=[], bkgd_img='', img_extent=[], experimenter='',
                  mouse_number='', mouse_sex = '', day='', trial='', entrance='', target=[],
                  time=[], r_nose=[], r_center=[], r_tail=[], filename=''):
       
        '''
        Metadata:
        -----------
        exp: str
            ID of experiment, also the date of when experiment started
            
        protocol_name: str
            name of experimental protocol
            
        protocol_description: str
            long str describing the protocol in plain words
            
        eth_file: int
            trial number corresponding to ethovision generated excel file
            
        bkgd_img: str
            name of the background image file
            
        img_extent: list
            coordinates of the background extent
            
        experimenter: str
            name of the experimenter who did this trial
            
        mouse_number: int
            ID number of the mouse
            
        mouse_sex: str
            male or female
            
        day: int
            day that trial was performed. Day 1 is experiment name
        
        trial: str
            number of training trials. if int, then it is regular training. Str could be habituation, probe, or reversal trial
            
        entrance: str
            direction mouse enters the arena from
            
        target: tuple
            coordinates of the target
            
        target_reverse: tuple, optional
            coordinates of reverse target
        
        Data:
        -----------        
            
        time: np.array seconds
            time since trial began
            
        r_nose: np array 2D
            x,y coordinates of nose points
            
        r_center: np array 2D
            x,y coordinates of center body points
            
        r_tail: np array
            x,y coordinates of base of tail points
            
            
        filename: str
            name of this trial as a mat file, initialized in the wrapper function
        ''' 
        
        # Store method parameters as attributes
        self.exp = exp
        self.protocol_name  = protocol_name
        self.protocol_description = protocol_description
        self.eth_file = eth_file
        self.bkgd_img = bkgd_img
        self.img_extent = img_extent
        self.experimenter = experimenter
        
        self.mouse_number = mouse_number
        self.mouse_sex = mouse_sex
        self.day = day
        self.trial = trial
        self.entrance = entrance
        self.target = target
        self.time = time
        self.r_nose = r_nose
        self.r_center = r_center
        self.r_tail = r_tail
        self.filename = filename
        
        
    def Store(self): #stores all experimental data and metadata as a mat file
        save_path = os.path.join(PROCESSED_FILE_DIR, self.exp, self.filename)
        
        #creates experiment directory if it does not already exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        #checks if save file already exists
        if os.path.exists(save_path):
            raise IOError(f'File {self.filename} already exists.')
        
        #create mat file here
        scipy.io.savemat(save_path,self.__dict__,long_field_names=True)
        
    def Load(self, exp, mouse, trial): #loads mat file

        try: path = glob.glob(glob.glob(PROCESSED_FILE_DIR+'/'+exp+'/')[0]+'*M%s_%s.mat'%(mouse, trial), 
                         recursive = True)[0] #finds file path based on ethovision trial number
        except: raise ValueError('The specified file does not exist ::: %s Mouse %s Trial %s'%(exp,mouse,trial))
        m = scipy.io.loadmat(path)
        self.exp = exp
        self.protocol_name=m['protocol_name'][0]
        self.protocol_description=m['protocol_description'][0]
        self.eth_file = m['eth_file'][0][0]
        self.bkgd_img=m['bkgd_img'][0]
        if isinstance(m['img_extent'], str): #if is string, convert to floats
            self.img_extent=[float(x) for x in m['img_extent'][0].split(',')]
        else: self.img_extent=m['img_extent'][0]
        self.experimenter=m['experimenter'][0]
        self.mouse_number=m['mouse_number'][0]
        self.mouse_sex=m['mouse_sex'][0]
        self.day=m['day'][0]
        self.trial=m['trial'][0]
        self.entrance=m['entrance'][0]
        self.target=m['target'][0]
        if params.check_reverse(self.exp, self.trial) is True:
            self.target_reverse=m['target_reverse'][0]
        self.time=m['time'][0]
        self.r_nose=m['r_nose']
        self.r_center=m['r_center']
        self.r_tail=m['r_tail']
        self.filename=m['filename'][0]
        if "velocity" in m: #checks if file has velocity info
            self.velocity = m['velocity'][0]
        if 'r_arena_holes' in m:
            self.r_arena_holes = m['r_arena_holes']
        if 'arena_circle' in m:
            self.arena_circle = m['arena_circle'][0]
        if 'k_reward' in m:
            self.k_reward = m['k_reward'][0][0]
        if 'k_hole_checks' in m:
            self.k_hole_checks = m['k_hole_checks']
        
    def Update(self):
        save_path = os.path.join(PROCESSED_FILE_DIR, self.exp, self.filename)
        
        #checks if save file already exists
        if not os.path.exists(save_path):
            raise IOError(f'File {self.filename} does not exist.')
            
        #update mat file here
        scipy.io.savemat(save_path,self.__dict__,long_field_names=True)
        
        

def get_excel_data(exp, eth_file):
    #import excel file based on experiment name and ethovision file
    fname = get_path_from_exp(exp, eth_file)
    
    def import_excel(fname, fname2):
        '''
        Imports excel file and stores it in the class TrialData
        '''
        
        print('     ... reading experiment %s excel file %i' %(exp,eth_file))
        
        #lesson learned: any hidden params in the init function are calculated with default values, they are not updated if these params are updated        
        t = TrialData(exp = exp, eth_file = eth_file)
        
        #accomadates version change in ethovision file, change header lengh if experiment was done after 2019
        if int((os.path.basename(os.path.dirname(fname)))[0:4]) > 2019: nrows_header = 39
        else: nrows_header = 37
        
        err_msg = ' *** ERROR  :::  %s data not found in ' + os.path.basename(fname)
        #opens trial header to get metadata
        h = pd.read_excel(fname,nrows=nrows_header,index_col=0,usecols='A,B')
        t.mouse_number = try_or_default(lambda: h.loc['Mouse Number'][0], default='', msg=err_msg%'Mouse Number' )
        t.day = try_or_default(lambda: h.loc['Day'][0], default='', msg=err_msg%'Day' )
        t.trial = try_or_default(lambda: h.loc['Trial'][0], default='', msg=err_msg%'Trial' )
        t.entrance = try_or_default(lambda: h.loc['Start Location'][0], default='', msg=err_msg%'Start Location' )
        
        t.filename = ('hfm_%s_M%s_%s.mat' %(t.exp, t.mouse_number, t.trial))
        
        #opens trial to get data
        d = pd.read_excel(fname,na_values=['-'],header=0, skiprows = nrows_header)
        t.time = try_or_default(lambda: d['Recording time'][1:].to_numpy().astype(float), default=np.array([]), msg=err_msg%'Recording time' )
        t.r_nose = try_or_default(lambda: d[['X nose','Y nose']][1:].to_numpy().astype(float), default=np.array([]), msg=err_msg%'X nose or Y nose' )
        t.r_center = try_or_default(lambda: d[['X center','Y center']][1:].to_numpy().astype(float), default=np.array([]), msg=err_msg%'X center or Y center' )
        t.r_tail = try_or_default(lambda: d[['X tail','Y tail']][1:].to_numpy().astype(float), default=np.array([]), msg=err_msg%'X tail or Y tail' )
        t.velocity = try_or_default(lambda: d['Velocity'][1:].to_numpy().astype(float), default=np.array([]), msg=err_msg%'Velocity' )
        
        #opens helper excel to get metadata
        i = pd.read_csv(fname2,header=0, index_col = 0)
        t.protocol_name = try_or_default(lambda: i.loc[exp,['Protocol']].to_numpy()[0], default='', msg=err_msg%'Protocol Name' )
        t.protocol_description = try_or_default(lambda: i.loc[exp,['Protocol Description']].to_numpy()[0], default='', msg=err_msg%'Protocol Description' )
        t.img_extent = try_or_default(lambda: np.array(i.loc[exp,['img_extent']][0].split(','),dtype=np.float64), default=np.array([]), msg=err_msg%'img_extent' )
        t.experimenter = try_or_default(lambda: i.loc[exp,['Experimenter']].to_numpy()[0], default='', msg=err_msg%'Experimenter' )
        t.mouse_sex = try_or_default(lambda: i.loc[exp,['mouse_sex']].to_numpy()[0], default='', msg=err_msg%'Mouse Sex excel')
        
        #determies data fromm parameter file
        t.bkgd_img = try_or_default(lambda: params.set_background_image(t.exp, params.check_reverse(t.exp, t.trial), t.entrance, t.trial), default='', msg=err_msg%'bkgd_img' )
        t.target = try_or_default(lambda: params.set_target(t.exp, t.entrance, t.trial), default=np.array([]), msg=err_msg%'Target' )
        if params.check_reverse(t.exp, t.trial) is True: #annotates false target, optional
            t.target_reverse = try_or_default(lambda: params.set_reverse_target(t.exp, t.entrance, t.trial), default=np.array([]), msg=err_msg%'Reverse Target' )
        if t.mouse_sex == 'mixed': 
            t.mouse_sex = try_or_default(lambda: params.get_mouse_sex(t.exp, t.mouse_number), default=np.array([]), msg=err_msg%'Mouse Sex mixed' )
        
        return t
    
    return import_excel(fname, DOCU)

def manual_single_excel_import(path, input_dict):
    
    print('     ... reading experiment excel file %s' %(path))
    
    exp = 'SingleTrial' #This determines the folder where the file is stored
    
    t = TrialData(exp = exp)
    
    t.eth_file = path.split('-')[-1]
    t.mouse_number = input_dict['mouse_number']
    t.mouse_sex = input_dict['mouse_sex']
    t.day = '1'#input('Enter Day')
    t.trial = input_dict['trial']
    t.entrance = 'na' #input('Enter entrance')
    t.protocol_name = input_dict['protocol_name']
    t.protocol_description = input_dict['protocol_description']
    t.img_extent = input_dict['img_extent']
    t.experimenter = 'Kelly'#input('enter experimenter name')
    
    t.bkgd_img = input_dict['bkgd_img']
    t.target = 'none'#input('enter target coordinates')
    t.filename = ('hfm_%s_M%s_%s.mat' %(t.exp, t.mouse_number, t.trial))
    
    #accomadates version change in ethovision file, change header lengh if experiment was done after 2019
    nrows_header = 36 #39 if all info is there, or 37 if before 2019
    
    err_msg = ' *** ERROR  :::  %s data not found in ' + os.path.basename(path)
    
    #opens trial header to get metadata NOT DONE
    # h = pd.read_excel(path,nrows=nrows_header,index_col=0,usecols='A,B')
    # t.mouse_number = try_or_default(lambda: h.loc['Mouse Number'][0], default=input('Enter mouse number'), msg=err_msg%'Mouse Number' )
    # t.day = try_or_default(lambda: h.loc['Day'][0], default='', msg=err_msg%'Day' )
    # t.trial = try_or_default(lambda: h.loc['Trial'][0], default='', msg=err_msg%'Trial' )
    
    #opens trial to get data
    d = pd.read_excel(path,na_values=['-'],header=0, skiprows = nrows_header)
    t.time = try_or_default(lambda: d['Recording time'][1:].to_numpy().astype(float), default=np.array([]), msg=err_msg%'Recording time' )
    t.r_nose = try_or_default(lambda: d[['X nose','Y nose']][1:].to_numpy().astype(float), default=np.array([0]), msg=err_msg%'X nose or Y nose' )
    t.r_center = try_or_default(lambda: d[['X center','Y center']][1:].to_numpy().astype(float), default=np.array([]), msg=err_msg%'X center or Y center' )
    t.r_tail = try_or_default(lambda: d[['X tail','Y tail']][1:].to_numpy().astype(float), default=np.array([0]), msg=err_msg%'X tail or Y tail' )
    t.velocity = try_or_default(lambda: d['Velocity'][1:].to_numpy().astype(float), default=np.array([]), msg=err_msg%'Velocity' )
    
    return t



'''for debugging'''

if __name__ == '__main__': 

    # objs = [TrialData() for i in range(2)]
    # objs[0].Load('2019-09-06', 10, '17')
    # objs[1].Load('2019-09-06', 10, 'R180 1')
    
    # datum = manual_single_excel_import(RAW_FILE_DIR + '\\2021-11-19_Raw Trial Data\Raw data-Hidden Food Maze-19Nov2021-Trial    79.xlsx')
    # datum.Store()
    
    data = TrialData()
    data.Load('SingleTrial', 'Nas1', '1')
    pass


    