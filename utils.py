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
Created on Wed Jul 18 16:44:41 2018

@author: gongotar
"""

import math
import random as rnd
import numpy as np

rnd_seed = None     # the random seed of the current simulation
Dauwe = None        # if Dauwe method is simulated
alternative_interval = None
lifetime_coefficient = 10 # the maximum expected lifetime compared to the computation time

def set_random_seed(seed):
    global rnd_seed
    rnd.seed(seed)
    rnd_seed = seed

def getNextFailure(lambd):
    """generates the time until next failure considering the given failure rate"""
    nextFailure = (-1 / lambd) * math.log(rnd.random())  
    return nextFailure

def determine_write_costs(system):
    """determine the checkpoint write cost for each level"""
    levels = system['levels']
    write_costs = len(levels) * [None]
    
    for i, level in enumerate(levels):
        lw = {}
        lw['level'] = level
        lw['w'] = level_write_cost(level, system)
        write_costs[i] = lw
    return write_costs

def determine_restart_costs(system):
    """determine the restart cost from each level"""
    levels = system['levels']
    restart_costs = len(levels) * [None]
    
    for i, level in enumerate(levels):
        lr = {}
        lr['level'] = level
        lr['r'] = level_restart_cost(level, system)
        restart_costs[i] = lr
    return restart_costs

def determine_levels_fail_rates(system, restart_costs, **extra):
    """determine the fail rate for each level"""
    levels = system['levels']
    fail_rates = (len(levels)+1) * [None]
    
    lf = {}
    lf['level'] = 0
    lf['f'] = level_fail_rate(0, system, restart_costs)
    fail_rates[0] = lf
    
    noh = 'no_hidden' in extra and extra['no_hidden']
    
    for i, level in enumerate(levels):
        lf = {}
        lf['level'] = level
        lf['f'] = level_fail_rate(level, system, restart_costs, no_hidden = noh)
        
#        fail_rates[i]['f'] = fail_rates[i]['f'] - lf['f']        # modified level failure rate
        
        fail_rates[i+1] = lf
    return fail_rates

def optimum_intervals(cp_write_costs, levels_fail_rates, levels_fail_rates_noh, yj):
    """determine the level optimum checkpoint intervals"""
    levels = [lw['level'] for lw in cp_write_costs]
    intervals = len(levels) * [None]
    we = len(levels) * [None]
    
    for i, level in enumerate(levels):
        li = {}
        li['level'] = level
        w = [lw['w'] for lw in cp_write_costs if lw['level'] == level][0]
        yl = [lf['f'] for lf in levels_fail_rates_noh if lf['level'] == levels[i]][0] # failure rate of the level (no hidden faults)
        if i > 0:
            y = [lf['f'] for lf in levels_fail_rates if lf['level'] == levels[i-1]][0]
            yw = yj - y - yl               # failure rate of cp write (failures not visible to the level and not considered by Daly Eq.)            
        else:
            y = [lf['f'] for lf in levels_fail_rates if lf['level'] == 0][0]
            yw = 0               # failure rate of cp write (failures not visible to the level and not considered by Daly Eq.)            

        if math.exp(w*yw) > 1:
            we[i] = {'level': level, 'we':(math.exp(w*yw)-1)/yw}   # expected duration of a successful write without any failures in other levels (not visible to this level)
        
        li['i'] = optimum_interval(y, w, yj, yl, i)
        intervals[i] = li
    return intervals, we

def fill_cps_interval(job, level_intervals, done_work, cp_write_costs):
    levels = [li['level'] for li in level_intervals]
    T = job['lifetime'] - done_work
    T = T*lifetime_coefficient
    lined = []
    for intv_pair in level_intervals: # fill checkpoints
        intv = intv_pair['i']
        level = intv_pair['level']
        computed = 0
        time = intv

        w = [lw['w'] for lw in cp_write_costs if lw['level']==level][0]
        while computed < T:
            cp = {'start': time, 'end': time+w, 'level': level, 'i': levels.index(level)}
            lined.append(cp)
            
            time += intv + w
            computed += intv

    lined.sort(key=lambda cp:cp['start'])
    return lined
    
    
def fill_cp_plan_interval_based(job, level_intervals, cp_write_costs, done_work, time, first_live_fail):
    """fill the job's plan regarding the checkpoint intervals (interval based scheduling)"""

    lined = fill_cps_interval(job, level_intervals, done_work, cp_write_costs)
    
    i = 0
    while i < len(lined) - 1: # remove overlapping checkpoints of different levels
                            # only keep the most stable chekcpoint
        if lined[i]['end'] > lined[i+1]['start']:
            if lined[i]['level'] > lined[i+1]['level']: # keep the most stable checkpoint
                del lined[i+1]
            else:
                del lined[i]
        else:
            i += 1
            
    
    job['plan']= lined
    
def fill_cp_plan_interval_based_all(job, level_intervals, cp_write_costs, done_work, time, first_live_fail):
    """fill the job's plan regarding the checkpoint intervals (interval based scheduling) write all levels cps in case of conflict"""

    lined = fill_cps_interval(job, level_intervals, done_work, cp_write_costs)
    # TODO: implement
    i = 0
    while i < len(lined) - 1: # remove overlapping checkpoints of different levels
                            # only keep the most stable chekcpoint
        if lined[i]['end'] > lined[i+1]['start']:
            if lined[i]['level'] > lined[i+1]['level']: # keep the most stable checkpoint
                del lined[i+1]
            else:
                del lined[i]
        else:
            i += 1
            
    
    job['plan']= lined
    
def fill_cp_plan_pattern_based(job, level_intervals, cp_write_costs, done_work, time, first_live_fail):
    """fill the job's plan regarding the checkpoint intervals (pattern based scheduling)"""

    global lifetime_coefficient
    
    levels = [li['level'] for li in level_intervals]
    T = job['lifetime'] - done_work
    T = T*lifetime_coefficient
    lined = []
    last_interval = 0
    for i, intv_pair in enumerate(level_intervals): # fill checkpoints
        level = intv_pair['level']
        w = [lw['w'] for lw in cp_write_costs if lw['level']==level][0]
        if i == 0:                                  # round the level intervals according to their proceding levels (if not first)
            intv = intv_pair['i']
            last_interval = intv + w
        else:
#            wu = [lw['w'] for lw in cp_write_costs if lw['level']==level_intervals[i-1]['level']][0] # upper level cp write
            prev_interval = last_interval
            no_of_prev_cps = max(int((intv_pair['i'])/(prev_interval)),2) # if no of previous level cps is less than 2 then use 2
            intv = no_of_prev_cps*prev_interval - w
            last_interval = intv + w
        computed = 0
        time = intv

        while computed < T:
            cp = {'start': time, 'end': time+w, 'level': level, 'i': levels.index(level)}
            lined.append(cp)
            
            time += intv + w
            computed += intv

    lined.sort(key=lambda cp:cp['start'])
    
    i = 0
    while i < len(lined) - 1: # remove overlapping checkpoints of different levels
                            # only keep the most stable chekcpoint
        if lined[i]['end'] > lined[i+1]['start']:
            if lined[i]['level'] > lined[i+1]['level']: # keep the most stable checkpoint
                del lined[i+1]
            else:
                del lined[i]
        else:
            i += 1
            
    
    job['plan']= lined
    
def fill_cp_plan(job, level_intervals, cp_write_costs, done_work, time, first_live_fail):
    """fill the job's plan regarding the checkpoint intervals (pattern based scheduling)"""
    
    lined = []
    levels = [li['level'] for li in level_intervals]
    levels_no = np.zeros(len(level_intervals))
    for idx, level in enumerate(levels):
        if idx < len(levels) - 1:
            i, opt = [(i,li['i']) for i,li in enumerate(level_intervals) if li['level'] == level][0]
            lower_opt = level_intervals[i+1]['i']
            w = [lw['w'] for lw in cp_write_costs if lw['level']==level][0]
            levels_no[idx] = int(lower_opt/(opt + w))
        elif idx == len(levels) - 1:
            levels_no[-1] = math.inf

    computed = 0
    from_last = 1
    most_stable = levels[-from_last]
    t = time
    last_cp_end = time
    
    T = job['lifetime'] - done_work
    T = T*6
    
    total_w = 0
    for level in levels:
        w = [lw['w'] for lw in cp_write_costs if lw['level']==level][0]
        lopt = [li['i'] for li in level_intervals if li['level'] == level][0]
        total_w += T*w/lopt

    while computed < T:
        opt = [li['i'] for li in level_intervals if li['level'] == most_stable][0]
        if computed == 0 and t + opt > T + total_w + time and most_stable != levels[0]:
            from_last += 1
            most_stable = levels[-from_last]
        else:
            if most_stable != levels[0]:
                computed, last_cp_end = gen_cp_rec(last_cp_end, computed, T, levels, level_intervals, levels[-from_last-1], levels_no, cp_write_costs, lined, first_live_fail)
            if computed < T:
                opt = [li['i'] for li in level_intervals if li['level'] == most_stable][0]
                t += opt
                computed += t - last_cp_end
            if computed < T:
                w = [lw['w'] for lw in cp_write_costs if lw['level']==most_stable][0]
                cp = {'start': t, 'end': t+w, 'level': most_stable, 'i': levels.index(most_stable)}
                lined.append(cp)
                t += w
                last_cp_end = t
        if t > first_live_fail:
            break

    if len(lined) > 0:
        lopt = [li['i'] for li in level_intervals if li['level'] == lined[-1]['level']][0]
        t = lined[-1]['end'] + lopt - (computed - T)
    else:
        t = T + time
    cp = {'start': t, 'end': t, 'level': most_stable, 'i': levels.index(most_stable)}
    lined.append(cp)
    job['plan']= lined

def determine_fail_level(system, restart_costs, first_live_fail, level_stack, no_restart_available_list):
    """determine the fail level considering fail time of nodes and job
    and defines the time of the next failures for the failed nodes/job"""
    
    time = first_live_fail['next_fail']
    levels = system['levels']
    nodes = system['nodes']
    fatals = []
    
    levels = [level for level in levels if level not in no_restart_available_list] # limit levels only to available ones
    
    available_level_stack = [level for level in level_stack if level not in no_restart_available_list] # limit level stack only to available levels
    
    if len(levels) == 0:
        return math.inf, system['A'], None, math.inf
    
    rt = [rc['r'] for rc in restart_costs if rc['level'] == levels[0]][0]

    if first_live_fail['next_fail'] == first_live_fail['next_trans_fail']:
        repair_time = time + rt
    elif first_live_fail['next_fail'] == first_live_fail['next_fatal_fail']:
        if len(levels) > 1:     # if beside local storage level, other levels are available to survive the work
            rf = [rc['r'] for rc in restart_costs if rc['level'] == levels[1]][0]
            repair_time = time + rf
        elif  levels[0] >= 4:   # if only one level is available which survives the node failure
            rf = [rc['r'] for rc in restart_costs if rc['level'] == levels[0]][0]
            repair_time = time + rf
        else:   # No levels to survive the work
            return math.inf, system['A'], None, math.inf
       
    processed = []
    fails_chain = [node for node in nodes if node['next_fail'] < repair_time]
    fails_chain.sort(key = lambda node:node['next_fail'])
    prev_chain = -1
    max_level = 1
    real_level = 1

    system['tmp'] = [(n['id'],n['next_fail']) for n in fails_chain]
    while len(fails_chain) > 0:
#        if system['l5'] > 0:
#            print(real_level,max_level,repair_time, len(fails_chain))
        processed.extend(fails_chain)
        if prev_chain == -1:
            prev_chain = 0
        else:
            prev_chain = repair_time - min(node['next_fail'] for node in fails_chain)
        delay = np.zeros(len(fails_chain))
        times = np.zeros(len(fails_chain))
        for i, node in enumerate(fails_chain):
            times[i] = node['next_fail']
            if node['next_fail'] == node['next_trans_fail']:    # node alive
                next_fail = getNextFailure(system['job_lambd'])
                while next_fail <= rt:
                    delay[i] += next_fail
                    next_fail = getNextFailure(system['job_lambd'])
                node['next_trans_fail'] += delay[i] + next_fail
                delay[i] += rt
            elif node['next_fail'] == node['next_fatal_fail']:  # node down
                level, real = process_fatal(system['bb'], fatals, levels, node, restart_costs,system, available_level_stack)  
                if (level == 5 or level == None) and 4 in levels:  # if failure at burst-buffer get next failure
                    system['bb']['next_fail'] += getNextFailure(system['bb_lambd'])
                    system['bb']['data'] = 0    # lost checkpoint data on burst buffer
                if level is not None:
                    max_level = max(max_level, level)
                    rf = [rc['r'] for rc in restart_costs if rc['level'] == level][0]
                else:   # no level available to restart the work, restart from beginning
                    max_level = math.inf
                    rf = 0
                if real is not None:
                    real_level = max(real_level, real)
                else:
                    real_level = math.inf
                next_fail = getNextFailure(system['node_lambd'])
                while next_fail <= rf:
                    delay[i] += next_fail
                    next_fail = getNextFailure(system['node_lambd'])
                node['next_fatal_fail'] += delay[i] + next_fail
                delay[i] += rf
                node['data'] = 0                # lost checkpoint data on the node
                node['xor_data'] = 0
                node['partner_data'] = 0
                fatals.append(node)
            node['old_fail'] = node['next_fail']
            node['next_fail'] = min(node['next_fatal_fail'], node['next_trans_fail'])


#        repair_time = max([sum(restart_end) for restart_end in zip(times, delay)]) + prev_chain     # the latest restart end defines the end repair time
        repair_time = max(repair_time, max([sum(restart_end) for restart_end in zip(times, delay)]))
        fails_chain = [node for node in nodes if node['next_fail'] < repair_time]
        fails_chain.sort(key = lambda node:node['next_fail'])

        
    if max_level == math.inf:
        return math.inf, system['A'], fatals, math.inf
    
    recover_level = max_level                               # find the most recent checkpoint level more stable than max level
    
    return recover_level, (repair_time - time), fatals, real_level

def perform_checkpoint(system, checkpoint, data, level_stack):
    """performs the checkpoint write and writes the given data 
    to the given level on nodes, bb or pfs"""

    level = checkpoint['level']
    nodes = system['nodes']
    
    if level == 1:                      # only local storage
        for node in nodes:
            node['data'] = data
    elif level == 2:                    # XOR level and local storage
        for node in nodes:
            node['data'] = data
            node['xor_data'] = data
    elif level == 3:                    # partner level and local storage
        for node in nodes:
            node['data'] = data
            node['partner_data'] = data
    elif level == 4:                    # only burst buffer
        system['bb']['data'] = data
        system['bb']['failed'] = False  # change the burst-buffer state to not failed
    elif level == 5:                    # only pfs
        system['pfs']['data'] = data
    else:
        raise ValueError('invalid level for performing checkpoint')
            
    # update the level stack (most recent level at the end)
    level_stack.remove(level)
    level_stack.append(level)

def process_fatal(bb, fails, levels, failed, restart_costs,system, available_level_stack):
    """analyses a set of fatal failures to determine the fatality level"""

    level = None
    if failed['next_fail'] == failed['next_trans_fail'] and 1 in levels:
        level = 1
    elif level == None:
        pass
    
    if level == None and 2 in levels:               # check if xor
        xor_id = failed['xor_group']
        r = [rc['r'] for rc in restart_costs if rc['level'] == 2][0]
        repair_time = failed['next_fail'] + r
        xor_group = [node for node in fails if node['xor_group'] == xor_id and node['old_fail'] < repair_time]
        if len(xor_group) == 0:     # if in the XOR group only one node is failed
            level = 2               # XOR
    elif level == None:
        pass

    if level == None and 3 in levels:               # check if partner
        partner_id = failed['partner']
        r = [rc['r'] for rc in restart_costs if rc['level'] == 3][0]
        repair_time = failed['next_fail'] + r
        partner = [node for node in fails if node['id'] == partner_id and node['old_fail'] < repair_time]
        prev_partner = [node for node in fails if node['partner'] == failed['id']]
        if len(partner) == 0 and len(prev_partner) == 0:       # if partner is not failed
            level = 3               # partner
    elif level == None:
        pass
    
    if level == None and 4 in levels:               # check if burst buffer
        r = [rc['r'] for rc in restart_costs if rc['level'] == 4][0]
        repair_time = failed['next_fail'] + r
        if repair_time < bb['next_fail'] and not bb['failed']:
            level = 4               # burst buffer
        else:
            bb['failed'] = True     # if repair time is after the bb failure, change the bb state
    elif level == None:
        pass

    if level == None and 5 in levels:               # if no other levels can restart the work use pfs
        level = 5                   # pfs
    elif level == None:
        pass
     
    # use the most recent level if it is more robust than the defined recover level
    recover_level = level
    
    if recover_level is not None:
        ind = len(available_level_stack) - 1
        while available_level_stack[ind] < recover_level:
            ind -= 1
        recover_level = available_level_stack[ind]
    
    return recover_level, level


def retrieve_level_checkpoint(system, level, fatals, level_stack):
    """load the most recent checkpoint from the given level (restart)"""
    
    nodes = system['nodes']
    computed = 0
    
    if level == 1:              # if level 1, no change to the node's data, simply return it (no failed nodes)
        return nodes[0]['data']
    elif level == 2:            # if level XOR, modify the data content of all nodes
        xor_data = []
        data = None
        i = 0
        for failed in fatals:
            i+=1
            x_group = failed['xor_group']
            # retrieve XOR data
            xor_data.extend([node['xor_data'] for node in nodes if node['xor_group'] == x_group and node['id'] != failed['id']])
            # check the retrieved data
            data = set(xor_data)
            if len(data) > 1:
                raise ValueError('different values retrieved from XOR group')
        if len(fatals) == 0:    # if node local level is the most recent level and fatals is empty
            xor_data = [nodes[0]['xor_data'],]  # every XOR data in every nodes can be used
        for node in nodes:
            node['data'] = xor_data[0]
            node['xor_data'] = xor_data[0]
        computed = xor_data[0]
    elif level == 3:            # if level partner, modify the data content of all nodes
        partner_data = None
        for failed in fatals:
            partner_id = failed['partner']
            p_data = [node['partner_data'] for node in nodes if node['id'] == partner_id][0]
            if partner_data == None:
                partner_data = p_data
            elif partner_data != p_data:
                raise ValueError('different values retrieved from partner scheme')
        if len(fatals) == 0:    # if node local/XOR level is the most recent level and fatals is empty
            partner_data = nodes[0]['partner_data']     # partner data on every node can be used
        for node in nodes:
            node['data'] = partner_data
            node['partner_data'] = partner_data
            node['xor_data'] = 0
        computed = partner_data
    elif level == 4:            # if level burst buffer, set all nodes to burst buffer data
        bb_data = system['bb']['data']
        for node in nodes:
            node['data'] = bb_data
            node['xor_data'] = 0
            node['partner_data'] = 0
        computed = bb_data
    elif level == 5:            # if level pfs, set all nodes to pfs data
        pfs_data = system['pfs']['data']
        for node in nodes:
            node['data'] = pfs_data
            node['xor_data'] = 0
            node['partner_data'] = 0
        computed = pfs_data
    else:
        for node in nodes:
            node['data'] = 0
            node['xor_data'] = 0
            node['partner_data'] = 0
        system['bb']['data'] = 0
        computed  = 0

    # update the level stack (most recent level at the end)
    if level != math.inf:
        level_stack.remove(level)
        level_stack.append(level)
        
    return computed

def level_write_cost(level, system):
    """for the given level returns the write cost"""
    node_rank = system['job']['node_rank']
    size = system['job']['cp_size']
    nsize = size*node_rank              # checkpoint size on each node
    nbw = system['nodes'][0]['wbw']     # node write bandwidth
    nbr = system['nodes'][0]['rbw']     # node read bandwidth
    net_r = system['net_r']             # network transfer rate
    
    if level == 1:      # local storage
        return nsize/nbw
    elif level == 2:    # XOR
        nx = system['xor_size']
        xor = system['xor_unit_comp']
        return nsize*(nx/(nbw*(nx - 1)) + 1/nbr + 2/net_r + xor)
    elif level == 3:    # partner
        return nsize*(2/nbw + 1/nbr + 2/net_r)
    elif level == 4:    # burst-buffer
        bbw = system['bb']['wbw']
        return len(system['nodes'])*nsize/bbw
    elif level == 5:    # pfs
        pfsw = system['pfs']['wbw']
        return len(system['nodes'])*nsize/pfsw
    else:               # invalid level
        return None
    
def level_restart_cost(level, system):
    """for the given level returns the restart cost"""
    node_rank = system['job']['node_rank']
    size = system['job']['cp_size']
    nsize = size*node_rank              # checkpoint size on each node
    nbw = system['nodes'][0]['wbw']     # node write bandwidth
    nbr = system['nodes'][0]['rbw']     # node read bandwidth
    net_r = system['net_r']             # network transfer rate
    A = system['A']                     # allocation delay
    
    if level == 1:      # local storage
        return nsize/nbr
    elif level == 2:    # XOR
        nx = system['xor_size']
        xor = system['xor_unit_comp']
        return A + nsize*((2*nx - 1)/(nbr*(nx - 1)) + 2/nbw + 2/net_r + xor)
    elif level == 3:    # partner
        return A + 3*nsize*(1/nbw + 1/nbr + 1/net_r)
    elif level == 4:    # burst-buffer
        bbr = system['bb']['rbw']
        return A + len(system['nodes'])*nsize/bbr
    elif level == 5:    # pfs
        pfsr = system['pfs']['rbw']
        return A + len(system['nodes'])*nsize/pfsr
    else:               # invalid level
        return None
    
def level_fail_rate(level, system, restart_costs, **extra):
    """for the given level returns the failure rate"""
    yt = system['node_lambd']
    n = system['job']['nodes_number']
    yj = system['job_fail_rate']

    if level == 0:      # job failures (node alive)
        return system['job_lambd']*n*system['job']['node_rank']
    elif level == 1:      # local storage
        return yt*n
    elif level == 2:    # XOR
        nx = system['xor_size']
        rx = [rc['r'] for rc in restart_costs if rc['level'] == 2][0]
#        yf = nx*(nx-1)*rx*(yt**2)*n/nx      # natural failure rate of XOR
        
        yf = n*(nx - 1)*(yt**2)*rx
        if 'no_hidden' in extra and extra['no_hidden']:
            return yf
        
        wx = level_write_cost(2, system)
        yu = level_fail_rate(1,         # the upper level failure rate
             system, 
             restart_costs) 
        
        
        tau = optimum_interval(yu, wx, yj, yf, 1)      # compute interval of XOR checkpointing
        nocp = (tau + wx)/2                             # wait time until cp
        yf = yf/(1 - yu*nocp)
        return yf 
    elif level == 3:    # partner
        rp = [rc['r'] for rc in restart_costs if rc['level'] == 3][0]
#        rp = system['A']
        yf = rp*yt**2*n                     # natural failure rate of PARTNER
#        if 2 in system['levels']:           # if XOR level is available
#            yu = level_fail_rate(2,         # the upper level failure rate
#                 system, 
#                 restart_costs) 
#            ydown = level_fail_rate(1,      # rate of havng a node down
#                         system, 
#                         restart_costs) 
#            wp = level_write_cost(3, system)    # get the cp write of the PARTNER level
#            tau = optimum_interval(yu, wp)      # compute interval of PARTNER checkpointing
#            yfp = yf +(tau+wp)*yu*ydown         # additional failures in case of a fault in XOR level before writing 
#                                                # a new PARTNER checkpoint after a PARTNER failure
#        else:
#            yfp = yf
        
        if 'no_hidden' in extra and extra['no_hidden']:
            return yf
            
        if 2 in system['levels']:           # if XOR level is available
            yx = level_fail_rate(2,         # the upper level failure rate
                         system, 
                         restart_costs) 
            yl = level_fail_rate(1,         # the upper level failure rate
                         system, 
                         restart_costs) 
            wp = level_write_cost(3, system)    # get the cp write of the PARTNER level
            
            tau = optimum_interval(yx, wp, yj, yf, 2)
            
            nocp = (tau + wp)/2                 # wait time until cp
            yfp = (yl*yx*nocp + yf)/(1-nocp*yx)
            
        else:
            yl = level_fail_rate(1,         # rate of upper level failure 
                 system, 
                 restart_costs)
            wp = level_write_cost(3, system)    # get the cp write of the PARTNER level

            
            tau = optimum_interval(yl, wp, yj, yf, 1)
            nocp = 0#(tau + wp)/2                 # wait time until cp
            yfp =  yf/(1-nocp*yl)

        return yfp 
    elif level == 4:    # burst-buffer
        ybl = system['bb_lambd'] # y_BB_lost rate
        index = system['levels'].index(level)
        if index > 0:   # if upper level checkpointing available
            ul = [l for i,l in enumerate(system['levels']) if i==index - 1][0]  # fetch the upper level
            ru = [rc['r'] for rc in restart_costs if rc['level'] == ul][0]      # upper level restart duration
            yu = level_fail_rate(ul, system, restart_costs)                     # upper level failure rate

            
        else:           # if burst-buffer is the most upper level
            ylj = level_fail_rate(0, system, restart_costs)                     # job failure rate (software)
            yln = level_fail_rate(1, system, restart_costs)                     # nodes failure rate (hardware)
            yu = ylj + yln                                                      # burst-buffer restores all software and hardware failures
            ru = 0                                                              # upper level restart duration

        
        wb = level_write_cost(level, system)
        
        tau = optimum_interval(yu, wb, yj, 0, index)             # checkpoint interval in the bb
        system['bb']['interval'] = tau
        rbb = system['A'] + wb + tau/2         # duration of having no checkpoints
        yfb = ybl*yu*(rbb + ru)
        return yfb
    elif level == 5:    # pfs
        return system['pfs_lambd']
    else:               # invalid level
        return None
    
def optimum_interval(y, w, yj, yl, level_index):
    """computes the optimum checkpointing interval for the given failure rate of data
    and checkpoint write duration"""
    global Dauwe, alternative_interval
    
    if not Dauwe and level_index > 0 and False:
            yw = yj - y - yl                # failure rate of cp write (failures not visible to the level and not considered by Daly Eq.)
            if math.exp(w*yw) > 1: # if failure rate are to not eligible
                w = (math.exp(w*yw)-1)/yw   # expected duration of a successful write without any failures in other levels (not visible to this level)
    
    MTTI = 1/y
    write_duration = w
    opt_int = MTTI;
    if MTTI > write_duration / 2:
        var = write_duration / (2 * MTTI);
        opt_int = math.sqrt(2 * MTTI * write_duration) * (1 + math.sqrt(var)/3 + var/9) - write_duration;

    if alternative_interval is not None and level_index == alternative_interval['level_index']:
        return alternative_interval['interval']
    
    return opt_int

def gen_cp_rec(last_cp_end, computed, T, levels, level_intervals, level, levels_no, cp_write_costs, lined, first_failure):
    t = last_cp_end
    for i in range(int(levels_no[levels.index(level)])):  
        if level != levels[0]:
            higher_level = levels[levels.index(level) - 1]
            computed, last_cp_end = gen_cp_rec(last_cp_end, computed, T, levels, level_intervals, higher_level, levels_no, cp_write_costs, lined, first_failure)
            if computed >= T:
                return computed, last_cp_end
        lopt = [li['i'] for li in level_intervals if li['level'] == level][0]
        t += lopt
        computed += t - last_cp_end
        if computed >= T:
            return computed, last_cp_end
        w = [lw['w'] for lw in cp_write_costs if lw['level']==level][0]
        cp = {'start': t, 'end': t+w, 'level': level, 'i': levels.index(level)}
        lined.append(cp)
        t += w
        last_cp_end = t
        
        if t > first_failure:
            return computed, last_cp_end
        
    return computed, last_cp_end

def update_unavailable_restart_levels(no_restart_available_list, levels, level, rl_level, activity):
    """updates the list of available levels for restart"""
    
    if activity == 'checkpointed':                      # if a checkpoint is performed
        if level in no_restart_available_list:          # the checkpointed level is avaiable for restart now
            no_restart_available_list.remove(level)
    elif activity == 'restarted' and rl_level > 1:      # if the job is restarted and at least one node is down
        no_restart_level = []
        if level == 3 and 2 in levels:                  # if the restarted level is partner (at least a node is failed)
            no_restart_level = [2]                      # XOR is not avaialble for checkpointing any more (XOR data removed from the failed nodes)
        elif level == 2 and 3 in levels:                # if the failed level is XOR (at least a node is failed)
            no_restart_level = [3]                      # partner data is removed from the failed nodes, so partner is not avaialable
        elif level >= 4:                                # if burst buffer is failed
            if 2 in levels:                             # some nodes were failed, means XOR data is lost on those nodes
                no_restart_level.append(2)
            if 3 in levels:                             # some nodes were failed, means partner data is lost on those nodes
                no_restart_level.append(3)
                
        for nrl in no_restart_level:                    # add the unavailable levels for restart to the list
            if nrl not in no_restart_available_list:
                no_restart_available_list.append(nrl)
    
def compute_xor_range(b, r, y, x):
    u = r*y
    down = (u - math.sqrt(u**2 + 4*u))/(2*u)
    up = (u + math.sqrt(u**2 + 4*u))/(2*u)
    return [max(down, (b*x - 2)/(b*x - 1)), up]
    
def expected_lifetime(y, T, R, w, opt):
    """computes the expecte lifetime of the given computation on single level"""
    MTTI = 1/y
    restart_time = R
    write_duration = w
    computation_time = T
    
    lifetime = MTTI * math.exp(restart_time / MTTI) \
            * (math.exp((opt + write_duration) / MTTI) - 1) * computation_time / opt
    return lifetime

def multilevel_expected_lifetime(data):
    """computes the expected lifetime of the job on multilevel"""
    
    system = data['system']
    level_rates = [d['f'] for d in data['levels_fail_rates']]
    level_intvs = [d['i'] for d in data['level_intervals']]
    level_costs = [d['w'] for d in data['cp_write_costs']]
    level_rests = [d['r'] for d in data['restart_costs']]
    levels = system['levels']
    job = system['job']
    T = job['lifetime']    
    
    for i,l in enumerate(levels):
        y = level_rates[i]
        R = level_rests[i]
        w = level_costs[i]
        opt = level_intvs[i]
        T = expected_lifetime(y, T, R, w, opt)
    
    return T
    
    
    
    
