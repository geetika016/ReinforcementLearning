#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 21:23:36 2018

@author: geetikasharma
"""

import numpy as np

import rl

def f(i):
    return (3 * G[i])/2 + np.random.normal()

def is_reward(signal):
    return signal > 11.85

def run(scheme,i2):
    LA = scheme()
    counts = np.zeros(8, dtype=int)
    for _ in range(i2):
        action = LA.get_action()
        counts[action] += 1
        LA.update(is_reward(f(action)))
    
    return counts

def test(scheme,runs = 100,i2=10000):
    format_string = '\033[36mn_{0}\033[0m = {1:6d}; \033[1mp_{0}\033[0m = {2:.6f}'

    counts = np.zeros(8, dtype=int)
    for _ in range(runs):
        counts += run(scheme, i2)

    for i in range(8):
        print(format_string.format(i, counts[i], counts[i]/(runs*i2)))


if __name__ == '__main__':
    G = [3, 5, 4, 2, 6, 8, 1, 7]
    #G = np.random.permutation([1,2,3,4,5,6,7,8])
    
    print('G = {}'.format(G))
    test(rl.LRI)
