#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:41:00 2017

mstarHF.py is a smaple script using the pyFTEGhf code to produce a plot
for the effective mass using the "thermodynamic route" proposed in

Phys. Rev. B ??, ?????? (2017)

in the Hartree-Fock approximation

author: Florian G. Eich
"""

'''
Use matplotlib to produce a pdf figure
'''
from matplotlib import pyplot as plt

plt.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'],
                 'monospace':['Computer Modern Typewriter']})
params = {'backend': 'pdf',
          'text.latex.preamble': [r"\usepackage{amsmath, amsfonts, amssymb}",
                                  r"\usepackage{dsfont}",
                                  r"\usepackage{color}"],
          'axes.labelsize': 10,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': [4.2, 3.15],
          'axes.unicode_minus': True}

plt.rcParams.update(params)

'''
include some standard libraries
'''
import sys
import numpy as np

import os.path

'''
we need to include the path to the 'src' directory so we can import
files from there ...
'''
sys.path.insert(0, "../src")

'''
import pyFTEGhf
'''
import pyFTEGhf as hfFTEG

'''
import the parameter file
'''
import params as PARAMS

'''
generating an array of densities
'''
rs = np.linspace(1.e-3, 1.e1, num=100, endpoint=True, retstep=False)
n = 3. / (4. * np.pi * rs**3)
'''
generating an array of temperatures
'''
# choose low temperature in terms of the Fermi temperature
theta = 1.e-3
TF = .5 * np.power(3. * np.pi**2 * n, 2. / 3. )

T = theta * TF

N = n
B = 1. / T


'''
compute the noninteracting electron gas
'''
PARAMS.C = 0.

niEG = hfFTEG.hfEG(beta=B, n=N)

# get heat capacity
cv0 = - niEG.beta**2 * niEG.dMHdNB[1,1] / niEG.n

'''
compute the interacting electron gas
'''
PARAMS.C = 1.

inEG = hfFTEG.hfEG(beta=B, n=N)

# get heat capacity
cv = - inEG.beta**2 * inEG.dMHdNB[1,1] / inEG.n

'''
m* plot as function of densities at fixed temperatures
determined by the ratio of the interacting and noninteracting heat capacity
'''
plt.ylabel(r'$m^\star / m$')
plt.xlabel(r'$r_\mathrm{s} [\mathrm{a.u.}]$')

plt.plot(rs, cv / cv0, lw=3, color='red', ls='-', \
label=r'$m^\star/m$')

C = 1. / (np.power(9. * np.pi / 4., 1. / 3.) * np.pi)
plt.plot(rs,1. / (1. - C * rs * np.log(theta / (4. * C * rs))), ls='--', lw=3, label="Bardeen's approximation")

plt.legend(loc=0)

figFile = 'mstarHF.pdf'
plt.savefig(figFile, bbox_inches = 'tight')
plt.close()
