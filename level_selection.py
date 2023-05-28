#   Copyright 2023 Zuse Institute Berlin
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:32:03 2019

@author: gongotar
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def expected_lifetime(y, T, R, w, opt):
    """computes the expected lifetime of the given computation on single level"""
    MTTI = 1/y
    restart_time = R
    write_duration = w
    computation_time = T
    
    lifetime = MTTI * math.exp(restart_time / MTTI) \
            * (math.exp((opt + write_duration) / MTTI) - 1) * computation_time / opt
    return lifetime

def optimum_interval(y, w):
    """computes the optimum checkpointing interval for the given failure rate of data
    and checkpoint write duration"""
    MTTI = 1/y
    write_duration = w
    optimum_interval = MTTI;
    if MTTI > write_duration / 2:
        var = write_duration / (2 * MTTI);
        optimum_interval = math.sqrt(2 * MTTI * write_duration) * (1 + math.sqrt(var)/3 + var/9) - write_duration;

    return optimum_interval

def multilevel_expected_lifetime(lifetime, level_rates, level_costs, level_rests, levels):
    """computes the expected lifetime of the job on multilevel"""

    T = lifetime
    
    for i,l in enumerate(levels):
        if i == 0:
            y = level_rates[0]
        else:
            y = level_rates[levels[i-1]]
        R = level_rests[l-1]
        w = level_costs[l-1]
        opt = optimum_interval(y, w)
        T = expected_lifetime(y, T, R, w, opt)

    # full restart failures from last level
    ll = levels[-1]
    lR = level_rests[ll-1]
    ly = level_rates[ll]
    
    try:
        T = math.exp(lR*ly)*(math.exp(T*ly)-1)/ly
    except OverflowError:
        T = math.inf
    
    return T

def gen_level(combination):
    bin_num = "{0:b}".format(combination).zfill(5)
    levels = []
    if bin_num[0] == '1':
        levels.append(1)
    if bin_num[1] == '1':
        levels.append(2)
    if bin_num[2] == '1':
        levels.append(3)
    if bin_num[3] == '1':
        levels.append(4)
    if bin_num[4] == '1':
        levels.append(5)
        
    return levels

def optimal_levels(lifetime, level_rates, level_costs, level_rests):
    combs = range(1, 32)
    levels = None
    min_time = math.inf
    for comb in reversed(combs):
        comb_levels = gen_level(comb)
        time = multilevel_expected_lifetime(lifetime, level_rates, level_costs, level_rests, comb_levels)
        if min_time > time:
            min_time = time
            levels = comb_levels
    return levels
    
lifetime = 100
level_rates = [0.4 ,0.2, 0.05, 0.005, 0.0005, 0.0001]
cost_range = range(3, 59, 1)
level1 = np.zeros(len(cost_range))
level2 = np.zeros(len(cost_range))
level3 = np.zeros(len(cost_range))
level4 = np.zeros(len(cost_range))
level5 = np.zeros(len(cost_range))

for i, cost in enumerate(cost_range):
    level_costs = [0.7, 2, cost, 55, 100]
    level_rests = [2*cost for cost in level_costs]
    levels = optimal_levels(lifetime, level_rates, level_costs, level_rests)
    if 1 in levels:
        level1[i] = 1
    if 2 in levels:
        level2[i] = 2
    if 3 in levels:
        level3[i] = 3
    if 4 in levels:
        level4[i] = 4
    if 5 in levels:
        level5[i] = 5


#  plot
    
f = plt.figure(figsize=(7, 5), dpi=80)
plt.plot(cost_range, level1)
plt.plot(cost_range, level2)
plt.plot(cost_range, level3)
plt.plot(cost_range, level4)
plt.plot(cost_range, level5)
plt.show()
