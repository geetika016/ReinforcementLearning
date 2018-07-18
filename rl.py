#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:35:18 2018

@author: geetikasharma

"""



import numpy as np

class LearningAutomaton:
    def __init__(self, N=4, k=8):
        self.N = k*N
        self.k = k
        self.state = 0
        self.depth = 0
    
    def get_action(self):
        return self.state
    
    def reward(self):
        pass
    
    def penalty(self):
        pass
    
    def update(self, reward):
        if reward:
            self.reward()
        else:
            self.punish()

    def __str__(self):
        return '({}/{}), ({}/{})'.format(self.state, self.k, self.depth, self.N)

class Tsetlin(LearningAutomaton):
    def reward(self):
        self.depth = min(self.depth + 1, self.N)
    
    def penalty(self):
        if self.depth == 0:
                self.state = (self.state + 1) % self.k
        else:
            self.depth -= 1

class Krinsky(Tsetlin):
    def reward(self):
        self.depth = self.N

class Krylov(Tsetlin):
    def penalty(self):
        if np.random.randint(2):
            self.reward()
        else:
            super().penalty()

class LRI:
    def __init__(self, states=8):
        self.n = states
        self.P = np.full(self.n, 1/self.n)
        self.alpha = 0.9
        self.prev_action = None


    def get_action(self):
        randomProb = np.random.random_sample()
        for anN in self.n:
            if(self.P > randomProb):
                self.prev_action = ((self.n),self.P)
            else:
                randomProb = randomProb - self.P
                
        #self.prev_action = np.random.choice(np.arange(self.n), p=self.P)
        return self.prev_action

    def reward(self):
        self.P *= self.alpha
        self.P[self.prev_action] += 1 - np.sum(self.P)

    def update(self, reward):
        if reward:
            self.reward()

    def __str__(self):
        return self.P


