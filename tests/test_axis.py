#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:55:29 2022

@author: krzysztof
"""

import unittest
import numpy as np

from salto import axis

class TestAxis(unittest.TestCase):

    def test_axis(self):
        A = np.array([-10, 2])
        B = np.array([5, 18])
        C = np.array([-6, 15])
        
        new_axis = salto.axis(A, B)
        
        C_scalar_projection = new_axis(C)
        
        points = [A, B, M, C]
        lines  = [[A, B]]
        labels = ['A', 'B', 'M', 'C']

