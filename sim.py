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
Created on Wed Jul 18 14:27:43 2018

@author: gongotar
"""
    
import utils as ut
import tests
from multiprocessing import Pool, cpu_count
from datetime import datetime
import os
import sys
import random as rnd
from mpi4py import MPI
import numpy as np
import traceback
import json
#from mpi_active_ranks import checkup_comm

# global units
h = 3600                # hour = 3600 seconds

# system and job properties
    
A = 60                  # allocation delay in seconds
net_r = 80              # network transfer rate in GB/s
xor_unit_comp = 0.01    # duration of computing XOR data of one GB checkpoints in seconds
nodes_numbr = 50000     # job's number of nodes
node_rank = 1           # job's rank per node
xor_size = 40000          # number of nodes in a XOR group
job_lambd = 1/(1000*h)  # job's failure rate in in 1/seconds per rank
node_lambd = 1/(8000*h)# node's failure rate in 1/seconds
bb_lambd = 1/(20000*h)   # burst-buffer's failure rate in 1/seconds
pfs_lambd = 0    # pfs's failure rate in 1/hours
lifetime = 20*h        # job's lifetime in seconds
cp_size = 118            # checkpoint size in GB of each rank
node_wbw = 15           # node write bandwidth in GB/s
bb_wbw = 20000            # burst-buffer write bandwidth in GB/s
pfs_wbw = 30000            # pfs write bandwidth in GB/s
node_rbw = 15           # node read bandwidth in GB/s
bb_rbw = 20000           # burst-buffer read bandwidth in GB/s
pfs_rbw = 30000           # pfs read bandwitdh in GB/s

levels = [1,3,4,5]        # used levels for checkpointing [1:local, 2:XOR, 3:partner, 4:burst-buffer, 5:pfs]
Single_test = True       # execute a single test (no parallel testing)
method = 'US_DW'        # failure rate computation method _ interval optimization method US: our method, 
                                                                                        #DW: dauwe (only interval optimization)
                                                                                        #SH: Sheng

# reassignment of parameters for the alternative case
#lifetime = 500*h
#node_wbw = 12
#node_rbw = 12
#pfs_wbw = 700
#prf_rbw = 1000
#job_lambd = 1/(8000*h)
#node_lambd = 1/(10000*h)

# extended for Dauwe
Dauwe = False            # use Dauwe method for planning checkpoints (2018), shit optimization, no failure rates
Sheng = {'interval': True, 'rates': False}           # use Cheng methdo for planning checkpoints (2014), wierd optimization, 
                                                        #failure rate (use optimization approach - failure rate computations)

Test_Modus = 2          # 1 = simple simulation, 2 = cp range simulation, 3 = interval range simulation

if node_lambd >= 1/(2000*h):
    ut.lifetime_coefficient = 60
    
def fill_failures(nodes, burst_buffer):
    """generate failure times for nodes and burst buffer"""
    
    burst_buffer['next_fail'] = ut.getNextFailure(bb_lambd)
    for node in nodes:
        node['next_fatal_fail'] = ut.getNextFailure(node_lambd)
        node['next_trans_fail'] = ut.getNextFailure(job_lambd)
        node['next_fail'] = min(node['next_fatal_fail'], node['next_trans_fail'])
        
def construct_env():

    job = {}
    burst_buffer = {}
    pfs = {}
    nodes = nodes_numbr*[None]
    
    job['nodes_number'] = nodes_numbr   # job's number of nodes
    job['lifetime'] = lifetime  # job's lifetime
    job['cp_size'] = cp_size    # job's cp size for each rank
    job['node_rank'] = node_rank# job's rank per node
    job['plan'] = []            # job's checkpoints plan
    job['prev_cp_end'] = 0      # end time of job's last checkpoint
    

    burst_buffer['data'] = 0
    burst_buffer['failed'] = False
    burst_buffer['rbw'] = bb_rbw
    burst_buffer['wbw'] = bb_wbw
    

    pfs['data'] = 0
    pfs['rbw'] = pfs_rbw
    pfs['wbw'] = pfs_wbw
    
    for i in range(nodes_numbr):
        node = {}
        node['id'] = i
        node['partner'] = (i+1) % nodes_numbr   # id of partner node
        node['xor_group'] = int(i/xor_size)
        node['data'] = 0
        node['partner_data'] = 0                # data of the partner
        node['xor_data'] = 0
        node['rbw'] = node_rbw
        node['wbw'] = node_wbw
        nodes[i] = node
        
    fill_failures(nodes, burst_buffer)          # generate failure times for nodes and burst buffer
    system = {}
    system['A'] = A
    system['net_r'] = net_r
    system['xor_unit_comp'] = xor_unit_comp
    system['job'] = job
    system['levels'] = levels
    system['nodes'] = nodes
    system['bb'] = burst_buffer
    system['pfs'] = pfs
    system['job_lambd'] = job_lambd
    system['node_lambd'] = node_lambd
    system['bb_lambd'] = bb_lambd
    system['pfs_lambd'] = pfs_lambd
    system['job_fail_rate'] = node_rank*nodes_numbr*job_lambd + nodes_numbr*node_lambd
    system['xor_size'] = xor_size
    
    return system
    
def construct_levels_properties(system):

    cp_write_costs = ut.determine_write_costs(system)                           # determine the checkpoint write cost for each level
    restart_costs = ut.determine_restart_costs(system)                          # determine the restart cost from each level
    
    # failure rate computations
    if Sheng['rates']:
        import utils_scheng as uts
        levels_eng_rates = uts.determine_engagement_rates(system, restart_costs)   # determine the fail rate for each level Scheng
        levels_fail_rates = uts.compatible_failure_rate(levels, levels_eng_rates)   # make engagement represtation compatible to ours
    else:
        levels_fail_rates = ut.determine_levels_fail_rates(system, restart_costs)   # determine the fail rate for each level
        
    
    # optimum interval computation
    if Dauwe:
        import utils_dauwe as utd
        delta = [w['w'] for w in cp_write_costs]
        R = [rc['r'] for rc in restart_costs]
        level_intervals = utd.lifetime_optimization(delta, 
                                                    levels_fail_rates, 
                                                    system['job_fail_rate'], 
                                                    R, 
                                                    len(system['nodes']), 
                                                    system['A'], 
                                                    system['job']['lifetime'])
    elif Sheng['interval']:
        import utils_scheng as uts 
        level_intervals = uts.optimum_intervals(cp_write_costs, levels_fail_rates, system['job']['lifetime'])
    elif not Sheng['interval']:
        levels_fail_rates_noh = ut.determine_levels_fail_rates(system, restart_costs, no_hidden=True)   # determine the fail rate for each level (no hidden failures)
        level_intervals, extended_costs = ut.optimum_intervals(cp_write_costs, levels_fail_rates, levels_fail_rates_noh,  system['job_fail_rate'])   # determine the level optimum checkpoint intervals
    
    data = {'cp_write_costs': cp_write_costs, 
            'restart_costs': restart_costs, 
            'levels_fail_rates': levels_fail_rates, 
            'level_intervals': level_intervals,
            'system': system}
    print(level_intervals)
    return data

#print(ut.compute_xor_range(node_wbw, [r['r'] for r in restart_costs if r['level']==2][0], node_lambd, xor_unit_comp))

def test(seed):
    """calls all necessary functions for performing a single test"""
    ut.set_random_seed(seed)
    system = data['system']
    fill_failures(system['nodes'], system['bb'])
    if len(sys.argv) == 1:  # if running in pc
        print('test number', seed, 'in process', os.getpid())

    time, fails_stats = tests.run_test(data)    # run the test and get the total time
    return time, fails_stats

# batch tests
    
if __name__ == '__main__' and len(sys.argv) == 1:  # pc
    
    if method[-2:] == 'DW':
        Dauwe = True
        Sheng['interval'] = False
    elif method[-2:] == 'SH':
        Dauwe = False
        Sheng['interval'] = True
    elif method[-2:] == 'US':
        Dauwe = False
        Sheng['interval'] = False
        
    if method[:2] == 'US':
        Sheng['rates'] = False
    elif method[:2] == 'SH':
        Sheng['rates'] = True
        
    ut.Dauwe = Dauwe
    # construct system and job
    system = construct_env()

    # job optimization computations
    data = construct_levels_properties(system)
    
    # mathematically expected lifetime
    exp_t = ut.multilevel_expected_lifetime(data)

    tests_no = cpu_count()*1    # number of tests
    total_time = 0              # total measured time for all tests
    seeds = [3*i for i in range(tests_no)]
    
    tests.init_test(method, data)        # initialize checkpoint plan for each level

    tick = datetime.now() # simulation start time
    if not Single_test:
        p = Pool(tests_no)
        res = p.imap_unordered(test, seeds)    

        for r in res:
            print(r)
            total_time += r[0]
        
        time = total_time/tests_no
    else:
        time, fails_stats = test(33529.0) 
            
    tock = datetime.now() # simulation end time
    dur1 = tock - tick    # simulation duration
    # report results
    
    print('simulation duration', dur1.total_seconds())    
    estimated = [float("%0.2f"%(lf['f']*time)) for lf in data['levels_fail_rates']]  # estimated number of failures during work
    
    print('Job performed in ', time, 's, ', time/h, 'h')
    print('Expected lifetime', exp_t)
    print('Actual level no. of failures ', fails_stats)
    print('Estimated level no. of failures ', estimated)
elif __name__ == '__main__' and Test_Modus == 1:                    # running on cluster (simple simulation)
    
    tests.print_info = False    # disable prints
    
    method = sys.argv[1]            # ME_PAT, ME_TIM, ME_TIM_ALL, DW_TIM
    lifetime = int(sys.argv[2])*h
    tests.report_interval = float(sys.argv[3])/float(sys.argv[2]) # define the reporting interval
    cp_size = int (sys.argv[4])
    nodes_numbr = int(sys.argv[5])
    
    if method[-2:] == 'DW':
        Dauwe = True
        Sheng['interval'] = False
    elif method[-2:] == 'SH':
        Dauwe = False
        Sheng['interval'] = True
    elif method[-2:] == 'US':
        Dauwe = False
        Sheng['interval'] = False
        
    if method[:2] == 'US':
        Sheng['rates'] = False
    elif method[:2] == 'SH':
        Sheng['rates'] = True
    
    ut.Dauwe = Dauwe
    
    # construct system and job
    system = construct_env()

    # job optimization computations
    data = construct_levels_properties(system)
      

    tests.init_test(method, data)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    
    tests.comm = comm

        
    # transfer random seed to all ranks
    ranks_seeds = np.zeros(ranks)
    if rank == 0:
        # mathematically expected lifetime
        exp_t = ut.multilevel_expected_lifetime(data)
        
        master_rand_seed = rnd.randint(0, 100000)   # master rand seed for this simulation
        rnd.seed(master_rand_seed)
        print("Master seed", master_rand_seed)
        
        for i in range(1, ranks):                   # send the rand seed to other ranks
            ranks_seeds[i] = rnd.randint(0, 100000)
            comm.send(ranks_seeds[i], dest=i, tag=1)
        rnd_seed = rnd.randint(0, 100000)           # get the rand seed of rank 0
        ranks_seeds[0] = rnd_seed
    else:
        rnd_seed = comm.recv(source=0, tag=1)       # receive the rank seed from rank 0
    ranks_seeds = comm.bcast(ranks_seeds, root=0)             # tell all ranks about seeds of all other ranks
       

    time = None
    try:
        time, fails_stats = test(rnd_seed)
    except Exception as ex:
        print('rank', rank,'with random seed', rnd_seed, 'faced error')
        print('Error message:')
        print(ex)
        print('')      
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        traceback.print_exc()
        print('\nEnvironment:')
        plan = system['job']['plan']
        cp = None
        if len(plan) > 0:
            cp = plan[0]
        print('\tplan len', len(plan),
              '\n\tnext checkpoint', cp,
              '\n\tjob prev cp', system['job']['prev_cp_end'])
    
    if rank == 0:
        times = np.zeros(ranks)
        times[0] = time
        for i in range(1, ranks):
            times[i] = comm.recv(source=i, tag=len(system['nodes']))
        
        print('simulation master random seed', master_rand_seed)
        print('expected lifetime', exp_t)
        print('avg lifetime',sum(times)/ranks)

        print('all lifetimes:')
        print(times)
    else:
        comm.send(time, dest=0, tag=len(system['nodes']))

elif __name__ == '__main__' and Test_Modus == 2:                    # running on cluster (cp range simulation)
    
    tests.print_info = False    # disable prints
    
    # get inputs
    method = sys.argv[1]            # ME_PAT, ME_TIM, ME_TIM_ALL, DW_TIM
    lifetime = int(sys.argv[2])*h
    tests.report_interval = float(sys.argv[3])/float(sys.argv[2]) # define the reporting interval
    nodes_numbr = int(sys.argv[4])
    levels_no = int(sys.argv[5])
    node_mttf = float(sys.argv[6])
    
    # set parameters
    node_lambd = 1/(node_mttf*h)
    
    if levels_no == 3:
        levels = [1,3,5]
    elif levels_no == 4:
        levels = [1,2,3,5]
    elif levels_no == 5:
        levels = [1,2,3,4,5]
    
    if method[-2:] == 'DW':
        Dauwe = True
        Sheng['interval'] = False
    elif method[-2:] == 'SH':
        Dauwe = False
        Sheng['interval'] = True
    elif method[-2:] == 'US':
        Dauwe = False
        Sheng['interval'] = False
        
    if method[:2] == 'US':
        Sheng['rates'] = False
    elif method[:2] == 'SH':
        Sheng['rates'] = True
    
    ut.Dauwe = Dauwe
    
    if Dauwe:
        ut.lifetime_coefficient = 60

    # test on nodes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    
    tests.comm = comm

    # transfer random seed to all ranks
    ranks_seeds = np.zeros(ranks)
    if rank == 0:
        master_rand_seed = rnd.randint(0, 100000)   # master rand seed for this simulation
        rnd.seed(master_rand_seed)
        print("Master seed", master_rand_seed)
        
        for i in range(1, ranks):                   # send the rand seed to other ranks
            ranks_seeds[i] = rnd.randint(0, 100000)
            comm.send(ranks_seeds[i], dest=i, tag=1)
        rnd_seed = rnd.randint(0, 100000)           # get the rand seed of rank 0
        ranks_seeds[0] = rnd_seed
    else:
        rnd_seed = comm.recv(source=0, tag=1)       # receive the rank seed from rank 0
    ranks_seeds = comm.bcast(ranks_seeds, root=0)             # tell all ranks about seeds of all other ranks
       
    cp_range = range(10, 66, 5)              # range of cp sizes to be tested
    cp_times = np.zeros(len(cp_range))  # different times for intervals
    
    if rank == 0:
        print('method', method)
        print('Dauwe', Dauwe)
        print('Sheng', Sheng)
        path = 'seeds'
        f = open(path, 'w+')
        json.dump(ranks_seeds.tolist(), f)
        f.close()
        
#    for j, size in enumerate(reversed(cp_range)):
    for j, size in enumerate(cp_range):
        cp_size = size
        data = None
            
        if rank == 0:
            print('testing cp size', size)
            system = construct_env()
            data = construct_levels_properties(system)
            tests.init_test(method, data)
            print('test initialized')
        data = comm.bcast(data, root=0)
        
        time = None
    
        try:
            time, fails_stats = test(rnd_seed)
        except Exception as ex:
            print('rank', rank,'with random seed', rnd_seed, 'and cp size', size, 'faced error')
            print('Error message:')
            print(ex)
        
        if rank == 0:
            times = np.zeros(ranks)
            times[0] = time
            for i in range(1, ranks):
                times[i] = comm.recv(source=i, tag=size)
#            cp_times[len(cp_range) - j - 1] = sum(times)/ranks
#            print('cp size', size, 'lifetime', cp_times[len(cp_range) - j - 1])
            cp_times[j] = sum(times)/ranks
            print('cp size', size, 'lifetime', cp_times[j])
        else:
            comm.isend(time, dest=0, tag=size)
    if rank == 0:
        print(cp_times)
        path = 'cps_data'
        f = open(path, 'w+')
        json.dump(cp_times.tolist(), f)
        f.close()
        print('all data written to file', path)

elif __name__ == '__main__' and Test_Modus == 3:                    # running on cluster (interval range test)
    
    tests.print_info = False    # disable prints
    
    method = sys.argv[1]            # ME_PAT, ME_TIM, ME_TIM_ALL, DW_TIM
    lifetime = int(sys.argv[2])*h
    tests.report_interval = float(sys.argv[3])/float(sys.argv[2]) # define the reporting interval
    cp_size = int (sys.argv[4])
    nodes_numbr = int(sys.argv[5])
    
    if method[-2:] == 'DW':
        Dauwe = True
        Sheng['interval'] = False
    elif method[-2:] == 'SH':
        Dauwe = False
        Sheng['interval'] = True
    elif method[-2:] == 'US':
        Dauwe = False
        Sheng['interval'] = False
        
    if method[:2] == 'US':
        Sheng['rates'] = False
    elif method[:2] == 'SH':
        Sheng['rates'] = True
    
    ut.Dauwe = Dauwe

    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranks = comm.Get_size()
    
    tests.comm = comm
    
    if rank == 0:
        print('mpi comm created')
        
                
    # construct system and job
    system = construct_env()

    # transfer random seed to all ranks
    ranks_seeds = np.zeros(ranks)
    if rank == 0:
        master_rand_seed = rnd.randint(0, 100000)   # master rand seed for this simulation
        rnd.seed(master_rand_seed)
        print("Master seed", master_rand_seed)
        
        for i in range(1, ranks):                   # send the rand seed to other ranks
            ranks_seeds[i] = rnd.randint(0, 100000)
            comm.send(ranks_seeds[i], dest=i, tag=1)
        rnd_seed = rnd.randint(0, 100000)           # get the rand seed of rank 0
        ranks_seeds[0] = rnd_seed
    else:
        rnd_seed = comm.recv(source=0, tag=1)       # receive the rank seed from rank 0
    ranks_seeds = comm.bcast(ranks_seeds, root=0)             # tell all ranks about seeds of all other ranks
        
        
    interval_range = range(7, 50)              # range of intervals to be tested
    intv_times = np.zeros(len(interval_range))  # different times for intervals
    
    for j, intv in enumerate(interval_range):
        ut.alternative_interval = {'level_index': 0, 'interval': intv}
        data = None
        if rank == 0:
            print('testing interval', intv)            
            # job optimization computations
            data = construct_levels_properties(system)
            # init the test
            tests.init_test(method, data)
            print('test initialized')
        data = comm.bcast(data, root=0)
        if rank == 0:
            print('data broadcasted')
        
        time = None


    
        try:
            time, fails_stats = test(rnd_seed)
        except Exception as ex:
            print('rank', rank,'with random seed', rnd_seed, 'and interval', intv, 'faced error')
            print('Error message:')
            print(ex)
        
        if rank == 0:
            times = np.zeros(ranks)
            times[0] = time
            for i in range(1, ranks):
                times[i] = comm.recv(source=i, tag=intv)
            intv_times[j] = sum(times)/ranks
            print('interval', intv, 'lifetime', intv_times[j])
        else:
            comm.send(time, dest=0, tag=intv)
    if rank == 0:
        path = 'intervals_data'
        f = open(path, 'w+')
        json.dump(intv_times.tolist(), f)
        f.close()
        print('all data written to file', path)
        print(intv_times)
