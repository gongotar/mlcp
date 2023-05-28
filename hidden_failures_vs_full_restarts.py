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
Created on Thu Dec 13 11:48:33 2018

@author: gongotar
"""

import math
import numpy as np

import matplotlib.pyplot as plt

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

def Tw(y, R, w, T, opt):
    MTTI = 1.0/y
    restart_time = R
    write_duration = w
    computation_time = T
    
    lifetime = MTTI * math.exp(restart_time / MTTI) \
            * (math.exp((opt + write_duration) / MTTI) - 1) * computation_time / opt
    return lifetime

h = 3600

Rb = 300
Rp = 100
Rx = 70
Rn = 20


Wb = 200
Wp = 50
Wx = 35
#Wn = 10

Tn = 200*h      # lifetime on nodes

yn = 1.0/(10000*h)    # node down rate
n = 10000       # node number
xg = 200        # XOR group size

XOR = True      # if XOR used
Partner = True  # if partner used

def test(Wb, Wp, Wx, yn, n):
    ############### full restarts
    
    if not XOR:
        Wx = 0
    if not Partner:
        Wp = 0
        
#    fRn = Rn        # restart duration from local
    fRb = Rb+Wp+Wx  # restart duration from burst buffer
    fRp = Rp+Wx     # restart duration from partner
    fRx = Wp+Rx     # restart duration from XOR
    
    yfl = n*yn                  # local failure rate
    yfx = n*(xg-1)*yn**2*fRx    # XOR failure rate
    yfp = n*yn**2*fRp           # partner failure rate
    
    # failures refering to each level
    y2x = yfl
    if XOR:
        y2p = yfx
    else:
        y2p = yfl
    
    if Partner:
        y2b = yfp
    elif XOR:
        y2b = yfx
    elif not XOR:
        y2b = yfl
    
    
    if XOR:
        opt1 = optimum_interval(y2x, Wx)
        Tw1 = Tw(y2x, fRx, Wx, Tn, opt1)
    else:
        Tw1 = Tn
    
    if Partner:
        opt2 = optimum_interval(y2p, Wp)
        Tw2 = Tw(y2p, fRp, Wp, Tw1, opt2)
    else:
        Tw2 = Tw1
        
    opt3 = optimum_interval(y2b, Wb)
    Tw3 = Tw(y2b, fRb, Wb, Tw2, opt3)
    
    
    ############### hidden failures
    if not XOR:
        Wx = 0
    if not Partner:
        Wp = 0
        
#    fRn = Rn        # restart duration from local
    fRb = Rb        # restart duration from burst buffer
    fRp = Rp        # restart duration from partner
    fRx = Wp        # restart duration from XOR
    
    yfl = n*yn                              # local failure rate
    optx = optimum_interval(yfl, Wx)
    yfx = n*(xg-1)*yn**2*fRx/(1-yfl*optx)    # XOR failure rate
    
    yfs = n*yn**2*fRp               # partner simultaneous failure rate
    
    # partner failure rate including hidden failures
    if XOR:
        optp = optimum_interval(yfx, Wp)
        yfp = (yfl*yfx*optp + yfs)/(1 - optp*yfx)
    else:
        optp = optimum_interval(yfl, Wp)
        yfp = yfs/(1 - optp*yfl)    
    
    # failures refering to each level
    y2x = yfl
    if XOR:
        y2p = yfx
    else:
        y2p = yfl
    
    if Partner:
        y2b = yfp
    elif XOR:
        y2b = yfx
    elif not XOR:
        y2b = yfl
    
    
    if XOR:
        opt1 = optimum_interval(y2x, Wx)
        hTw1 = Tw(y2x, fRx, Wx, Tn, opt1)
    else:
        hTw1 = Tn
    
    if Partner:
        opt2 = optimum_interval(y2p, Wp)
        hTw2 = Tw(y2p, fRp, Wp, hTw1, opt2)
    else:
        hTw2 = hTw1
        
    opt3 = optimum_interval(y2b, Wb)
    hTw3 = Tw(y2b, fRb, Wb, hTw2, opt3)
    
    return Tw3, hTw3

items = range(8000, 50000, 200)

fTw = np.zeros(len(items))
hTw = np.zeros(len(items))

for i,d in enumerate(items):
    fTw[i], hTw[i] = test(Wb, Wp, Wx, yn, d)
    
#print(test(Wb, Wp, Wx, yn, n))
f = plt.figure(figsize=(7, 5), dpi=80)
plt.plot(items, fTw, color='green')
plt.plot(items, hTw, color='red')
plt.show()
