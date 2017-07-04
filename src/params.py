#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 20:37:47 2014

params.py holds various definitions and parameters for pyFTEGhf.py

author:
    Florian G. Eich

changelog:
    * 30-06-2017: clean-up for initial gitHub commit
    
"""

import numpy as np

'''
Numerical constants
'''

# infinitesimal ~ 10^(-12) for double precision
eta = 1.e4 * np.finfo(np.complex_).eps

# tiny ~ 10^(-9) for double precision
tiny = 1.e3 * eta

# small ~ 10^(-6) for double precision
small = 1.e3 * tiny

# delta ~ 10^(-3) for double precision
delta = 1.e3 * small

'''
Code parameters for pyFTEGhf.py
'''
# maximum number of iterations in self-consistency cycle
nSC=500

# number of integration mesh points given by 2^(M)
M = 6

# coupling constant
C = 1.

# convergence criterion for inversion
dq = tiny

# maximum number of iterations for density-potential inversion
maxIterPotInv = 100

# convergence criterion for density-potential inversion
relErrPotInv = tiny

# debug flag that triggers additional output
debug = False

