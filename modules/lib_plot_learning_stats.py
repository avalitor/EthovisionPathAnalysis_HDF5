# -*- coding: utf-8 -*-
"""
Created on Fri May 20 02:12:43 2022

plots learning data like latency, distance, and speed
also plots search bias using dwell time around target
gets data from calc_latency_distance_speed

@author: Kelly
"""

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import curve_fit #for curve fitting
import scipy.signal #for smooth curve
import scipy.stats as stats
from modules.calc_latency_distance_speed import iterate_all_trials, calc_search_bias
from modules.config import ROOT_DIR

'''
Helper Functions for graphing
'''
def exponential_regression(xs, ys):
    '''Calculates the exponential best fit line of x and y coordinates'''
    #exponential regression
    def monoExp(x, m, t, b):
        return m * np.exp(-t * x) + b
    
    p0 = (2000, 0.1, 50) # start with values near those we expect
    params, cv = curve_fit(monoExp, xs, ys, p0)
    m, t, b = params
    y_fit =  monoExp(xs, m, t, b)
    # print(f"Y = {m} * e^(-{t} * x) + {b}") #inspect function coefficients
    
    # determine quality of the fit
    squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
    squaredDiffsFromMean = np.square(ys - np.mean(ys))
    r_squared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    return y_fit, np.round(r_squared, 4)

def linear_regression(xs, ys):
    def squared_error(ys_orig,ys_line):
        return sum((ys_line - ys_orig) * (ys_line - ys_orig))

    #calculates the R^2 that shows how well the line fits
    def coefficient_of_determination(ys_orig,ys_line):
        y_mean_line = [np.mean(ys_orig) for y in ys_orig]
        squared_error_regr = squared_error(ys_orig, ys_line)
        squared_error_y_mean = squared_error(ys_orig, y_mean_line)
        return 1 - (squared_error_regr/squared_error_y_mean)
    
    #calculates the best fit line using the polyfit function
    theta = np.polyfit(xs, ys, 1)
    y_line = theta[1] + theta[0] * xs
    
    r_squared = coefficient_of_determination(ys, y_line)
    return y_line, np.round(r_squared, 4)

def smooth_curve(xs, ys):
    # apply a 3-pole lowpass filter at 0.1x Nyquist frequency
    b, a = scipy.signal.butter(3, 0.2) #10# cutoff frequency plus lowpass filter
    filtered = scipy.signal.filtfilt(b, a, ys, method="gust") #uses Gustafsson’s Method
    return filtered
    

'''
Plotting Functions
'''
def plot_latency(data, bestfit = False, log = True, savefig = False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    
    x = data['Latency'].index
    y = data['Latency']

    avg = np.nanmean(y, 1)
    SE = np.nanstd(y, 1)/np.sqrt(y.shape[1])
    
    if bestfit: #try to create line of best fit
        try:
            #exponential regression
            y_line, r_squared = exponential_regression(x, avg)
            fit = "R² = %s"%r_squared
            line1, = ax.plot(x, y_line, '--', color='#004E89', label=fit)
            
        except:
            #linear regression
            y_line, r_squared = linear_regression(x, avg)
            fit = "R² = %s"%r_squared
            line1, = ax.plot(x, y_line, '--', color='#004E89', label=fit)
    
    # filtered = smooth_curve(x, avg)
    # ax.plot(x, filtered, '-', color='#004E89', label=fit)
    else: line1, = ax.plot(x, avg, color='#004E89', label='Avg') #plots average
    # line1 = ax.errorbar(x, avg, yerr=SE, color='#004E89', label='Avg') #plots average
    
    fill1 = ax.fill_between(x, np.nanmin(y, axis=1), np.nanmax(y, axis=1), alpha=0.25, color='#004E89', ec='None', label='Range')
    fill2 = ax.fill_between(x, avg+SE, avg-SE, alpha=0.25, color='#004E89', ec='None', label='SE') #standard error
    
    
    ax.set_xlabel('Trials', fontsize=13)
    ax.set_ylabel('Latency (s)', fontsize=13)
    
    leg3 = patches.Patch(alpha=0.5, color='#004E89', ec='None', label=fill2.get_label())
    ax.legend(handles=[line1, fill1, leg3], fontsize=11, loc='upper right') #legend for average line, upper or lower
    
    #annotate R^2 value
    # ax.annotate(r_square, xy=(12, np.nanmin(avg)), xycoords='data',
    #             xytext=(0, 0), textcoords='offset points')
    
    ax.grid(False) #hide gridlines
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
#    ax.spines['left'].set_position('zero')
#    ax.spines['bottom'].set_position('zero')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) #sets x-axis spacing
    if log:
        ax.set(yscale="log") #set a logarithmic scale
    
    if savefig == True:
            plt.savefig(ROOT_DIR+'/figures/AvgLatency M%s-%s.png'%(data['Latency'].columns[0], data['Latency'].columns[-1]), dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()


def plot_distance(data, bestfit = False, log = True, savefig = False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    ax.set_position([0.125, 0.125, 0.72, 0.704])
    
    x = data['Distance'].index
    y = data['Distance']

    avg = np.nanmean(y, 1)
    SE = np.nanstd(y, 1)/np.sqrt(y.shape[1])
    
    if bestfit:
        try:
            #exponential regression
            y_line, r_squared = exponential_regression(x, avg)
            fit = "R² = %s"%r_squared
            line1, = ax.plot(x, y_line, '--', color='#5F0F40', label=fit)
            
        except:
            #linear regression
            y_line, r_squared = linear_regression(x, avg)
            fit = "R² = %s"%r_squared
            line1, = ax.plot(x, y_line, '--', color='#5F0F40', label=fit)
    
    else: line1, = ax.plot(x, avg, color='#5F0F40', label='Avg') #plots average
    
    # line1 = ax.errorbar(x, avg, yerr=SE, color='#5F0F40', label='Distance')
    fill1 = ax.fill_between(x, np.nanmin(y, axis=1), np.nanmax(y, axis=1), alpha=0.25, color='#5F0F40', ec='None', label='Range') #range
    fill2 = ax.fill_between(x, avg+SE, avg-SE, alpha=0.25, color='#5F0F40', ec='None', label='SE') #standard error
    
    ax.set_xlabel('Trials', fontsize=13)
    ax.set_ylabel('Distance (cm)', fontsize=13)
    
    leg4 = patches.Patch(alpha=0.5, color='#5F0F40', ec='None', label=fill2.get_label())
    ax.legend(handles=[line1, fill1, leg4], fontsize=11, loc='upper right') #legend for average line
    
    ax.grid(False) #hide gridlines
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) #sets x-axis spacing
    if log:
        # ax.set(yscale="log") #set a logarithmic scale
        ax.set_yscale('log') #set a logarithmic scale
    
    if savefig == True:
            plt.savefig(ROOT_DIR+'/figures/AvgDistance M%s-%s.png'%(data['Distance'].columns[0], data['Distance'].columns[-1]), dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    
def plot_speed(data, bestfit = False, log = False, savefig = False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
    
    x = data['Speed'].index
    y = data['Speed']

    avg = np.nanmean(y, 1)
    SE = np.nanstd(y, 1)/np.sqrt(y.shape[1])
    
    if bestfit:
        try:
            #exponential regression
            y_line, r_squared = exponential_regression(x, avg)
            fit = "R² = %s"%r_squared
            line1, = ax.plot(x, y_line, '--', color='#F18701', label=fit)
            
        except:
            #linear regression
            y_line, r_squared = linear_regression(x, avg)
            fit = "R² = %s"%r_squared
            line1, = ax.plot(x, y_line, '--', color='#F18701', label=fit)
    else: line1, = ax.plot(x, avg, color='#F18701', label='Avg') #plots average
    
    # line1 = ax.errorbar(x, avg, yerr=SE, color='#F18701', label='Speed')
    fill1 = ax.fill_between(x, np.nanmin(y, axis=1), np.nanmax(y, axis=1), alpha=0.25, color='#F18701', ec='None', label='Range') #range
    fill2 = ax.fill_between(x, avg+SE, avg-SE, alpha=0.25, color='#F18701', ec='None', label='SE') #standard error
    
    ax.set_xlabel('Trials', fontsize=13)
    ax.set_ylabel('Speed (cm/s)', fontsize=13)
    
    leg4 = patches.Patch(alpha=0.5, color='#F18701', ec='None', label=fill2.get_label())
    ax.legend(handles=[line1, fill1, leg4], fontsize=11, loc='upper left') #legend for average line
    
    ax.grid(False) #hide gridlines
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) #sets x-axis spacing
    
    if log:
        ax.set(yscale="log") #set a logarithmic scale
    
    if savefig == True:
            plt.savefig(ROOT_DIR+'/figures/AvgSpeed M%s-%s.png'%(data['Speed'].columns[0], data['Speed'].columns[-1]), dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    
def plot_compare_curves(data1, data2, label1, label2, show_sig = True, log = False, savefig = False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    
    y1 = data1.dropna(axis=0)
    y2 = data2.dropna(axis=0)
    
    if len(y1) != len(y2): #if they are not the same length
        #find min axis and crop
        y1 = data1.iloc[:min(len(y1), len(y2))]
        y2 = data2.iloc[:min(len(y1), len(y2))]
    
    x = y1.index

    avg1 = np.nanmean(y1, 1)
    avg2 = np.nanmean(y2, 1)
    SE1 = np.nanstd(y1, 1)/np.sqrt(y1.shape[1])
    SE2 = np.nanstd(y2, 1)/np.sqrt(y2.shape[1])
    
    
    line1 = ax.errorbar(x, avg1, yerr=SE1, color='#004E89', label=label1)
    line2 = ax.errorbar(x, avg2, yerr=SE2, color='#C00021', label=label2)
    
    if show_sig:
        t_test = stats.ttest_ind(y1, y2, axis=1, nan_policy = 'omit')[1]
        
        def convert_pvalue_to_asterisks(pvalue):
            if pvalue <= 0.0001:
                return "****"
            elif pvalue <= 0.001:
                return "***"
            elif pvalue <= 0.01:
                return "**"
            elif pvalue <= 0.05:
                return "*"
            return " "
        
        pvalue_asterisks = []
        for p in t_test:
            stars = convert_pvalue_to_asterisks(p)
            pvalue_asterisks.append(stars)
        
        y_position = np.maximum(np.nanmean(y1, 1), np.nanmean(y2,1))*1.2
        # y_position = y2.max(axis=1)
        for idx, pval in enumerate(pvalue_asterisks):
            # plt.text(x=idx+1, y=y_position, s=pval)
            ax.annotate(pval, (x[idx], y_position[idx]))
            print(f'Trial {x[idx]} is {pval}')
    
    ax.set_xlabel('Trials', fontsize=13)
    ax.set_ylabel('Speed (cm/s)', fontsize=13)
    
    ax.legend(handles=[line1, line2], fontsize=11, loc='lower right') #legend for average line
    
    # ax.set_ylim(5, 40)
    ax.grid(False) #hide gridlines
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2)) #sets x-axis spacing
    
    if log:
        ax.set(yscale="log") #set a logarithmic scale
    
    if savefig == True:
            plt.savefig(ROOT_DIR+f'/figures/CompareCurves_{label1}_vs_{label2}.png', dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()

def plot_percent_bar(data, savefig= False):
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    start = 0
    Q1 = (data[0]*100)/np.sum(data)
    Q2 = (data[1]*100)/np.sum(data)
    Q3 = (data[2]*100)/np.sum(data)
    Q4 = (data[3]*100)/np.sum(data)
    
    ax.broken_barh([(start, Q1), (Q1, Q2), (Q1+Q2, Q3), (Q1+Q2+Q3, Q4)], [0.3, 2], 
                   edgecolor = 'k', facecolors=('#C00021', '#ffffff', '#ffffff', '#ffffff'))
    ax.set_ylim(0, 30)
    ax.set_xlim(0, 101)
    
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.axis('off')
     
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    
    ax.set_axisbelow(True) 
    
    # ax.set_yticklabels(['Q1'])
    # ax.grid(axis='x')
    ax.set_xlabel('% Time', fontsize=12)
    ax.text(Q1/2, 3, 'Q1', fontsize=12, ha = 'center')
    ax.text((Q1+Q2)-Q2/2, 3, "Q2", fontsize=12, ha = 'center' )
    ax.text((Q1+Q2+Q3)-Q3/2, 3, 'Q3', fontsize=12, ha = 'center')
    ax.text((Q1+Q2+Q3+Q4)-Q4/2, 3, 'Q4', fontsize=12, ha = 'center')
    
    # fig.suptitle('Percent Bar', fontsize=16)
    
    # leg1 = mpatches.Patch(color='#ffffff', label='Q1')
    # leg2 = mpatches.Patch(color='#ffffff', label='D2')
    # leg3 = mpatches.Patch(color='#C00021', label='Q3')
    # leg4 = mpatches.Patch(color='#C00021', label='Q4')
    # ax.legend(handles=[leg1, leg2, leg3, leg4], ncol=4)
    
    if savefig == True:
            plt.savefig(ROOT_DIR+'/figures/PercentBar.png', dpi=600, bbox_inches='tight', pad_inches = 0)
    
    plt.show()

if __name__ == '__main__':
    # static = iterate_all_trials(['2019-05-23','2022-11-04'], continuous= False)
    # with_cue, no_cue = static['Distance'][['5','6', '7', '8']], static['Distance'][['1','2', '3', '4']]
    # plot_compare_curves(with_cue, no_cue, 'With Cues', 'No Cues', log = False)
    
    # dark_trial = iterate_all_trials(['2019-06-18','2022-11-04'], training_trials_only = False, continuous= False)
    # dark, light = dark_trial['Distance'][['5','6','7','8']], dark_trial['Distance'][['1','2','3','4']]
    # plot_compare_curves(dark, light, 'Trained in Light, Trial in Dark', 'Trained in Light', show_sig = False, log = False)
    
    # sex_trial = iterate_all_trials(['2022-08-12','2022-09-20'], training_trials_only = True, continuous= False)
    # male, female = sex_trial['Speed'][['69','70','71','72']], sex_trial['Speed'][['73','74','75','76']]
    # plot_compare_curves(male, female, 'Male', 'Female', show_sig = True, log = False)
    pass