#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 12:34:58 2018

@author: geetikasharma
"""

import colorama
colorama.init()

import numpy as np

import rl

def f(i):
    return 1.5 * G[i] + np.random.normal()

def is_reward(signal):
    return signal > 10

def run(scheme, batch_size, cutoff):
    LA = scheme()
    for _ in range(batch_size - cutoff):
        action = LA.get_action()
        LA.update(is_reward(f(action)))

    counts = np.zeros(8, dtype=int)
    for _ in range(cutoff):
        action = LA.get_action()
        counts[action] += 1
        LA.update(is_reward(f(action)))
    
    return counts

def test(scheme=rl.Tsetlin, batch_number=100, batch_size=15000, cutoff=5000):
    # for pretty colours
    format_string = '\033[34mn_{0}\033[0m = {1:6d}; \033[34mp_{0}\033[0m = {2:.3f}'

    counts = np.zeros(8, dtype=int)
    for _ in range(batch_number):
        counts += run(scheme, batch_size, cutoff)

    for i in range(8):
        print(format_string.format(i, counts[i], counts[i]/(batch_number*cutoff)))

if __name__ == '__main__':
    # G = np.random.permutation(8)
    G = [6, 1, 4, 2, 7, 0, 3, 5]
    print('G = {}'.format(G))

    print('Testing Tsetlin...')
    test(rl.Tsetlin)

    print('Testing Krinsky...')
    test(rl.Krinsky)

    print('Testing Krlyov...')
    test(rl.Krylov)

    print('Testing L_RI...')
    test(rl.LRI)