# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:44:32 2022

Makes percent bars for static and changing entrances
for spatial learning manuscript

@author: Kelly
"""

import modules.lib_plot_learning_stats as ls
import modules.calc_latency_distance_speed as calc

'''
PLOT PERCENT BAR
*****************
    
Rotating Entrances
'''
# d = calc.calc_search_bias(['2021-07-16', '2021-11-15'], '2min')
# ls.plot_percent_bar(d)

'''
Static Entrances
'''
# d = calc.calc_search_bias(['2019-09-06', '2019-10-07'], '2min')
# ls.plot_percent_bar(d)


'''
LATENCY, DISTANCE, SPEED LEARNING CURVES
***************************************

Rotating & Static Entrances
'''
# rotate = calc.iterate_all_trials(['2021-07-16', '2021-11-15'], continuous= False)
# ls.plot_latency(rotate, log=True, savefig = False)
# ls.plot_distance(rotate, log=True, savefig = False)
# ls.plot_speed(rotate, savefig = False)
# calc.curve_pValue(rotate)

static = calc.iterate_all_trials(['2019-09-06','2019-10-07'], continuous= False)
ls.plot_latency(static, log=True, savefig = False)
ls.plot_distance(static, log=True, savefig = False)
ls.plot_speed(static, savefig = False)
calc.curve_pValue(static)

'''
3 Local Cues
'''
# loc = calc.iterate_all_trials(['2019-12-11','2021-08-11'], continuous= False)
# ls.plot_latency(loc, log=True, savefig = False)
# ls.plot_distance(loc, log=True, savefig = False)
# ls.plot_speed(loc, savefig = False)
# calc.curve_pValue(loc)
