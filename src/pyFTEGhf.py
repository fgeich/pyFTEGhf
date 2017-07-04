#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:38:14 2017

pyFTEGhf.py provides a class representing the interacting electron gas
at finite temperature within self-consistent Hartree-Fock theory.

author:
    Florian G. Eich

changelog:
    * 30-06-2017: clean-up for initial gitHub commit
    
"""

import numpy as np
from numpy import linalg as LA
from scipy import optimize as OPT
from numpy import polynomial as POLY

import params as PARAMS

class hfEG:
    """
    Class representing the electron gas at finite temperature in
    the self-consistent Hartree-Fock approximation.
    """

    def __init__(self, alpha=None, beta=None, n=None, h=None):
        '''
        initialization requires:

        alpha -> relative dimensionless chemical potential (optional)
        beta  -> inverse temperature (optional)
        n     -> density (optional)
        h     -> energy density (optional)

        Two out of the four quantities have to be specified
        
        Comments:
        * Currently only the pairs (alpha, beta) and (n, beta) implemented.
        
        '''      
        # set coupling constant
        self.C = PARAMS.C
        # set maximum number of iterations in self-consistent cycle
        self.nSC = PARAMS.nSC
        # set number of integration points
        N = 2**PARAMS.M

        # alpha and beta given
        if alpha is not None and beta is not None and n is None and h is None:
            if alpha.shape != beta.shape:
                print "# (pyFTEGhf:init) Error: alpha and beta given, but do not have the same shape!"
                print
    
                return
    
            # store shape
            shape = alpha.shape
    
            alpha = alpha.flatten()
            beta = beta.flatten()            
            
            self.A = alpha
            self.B = beta            

        # beta and n given
        elif alpha is None and beta is not None and n is not None and h is None:
            if n.shape != beta.shape:
                print "# (pyFTEGhf:init) Error: n and beta given, but do not have the same shape!"
                print
    
                return
    
            # store shape
            shape = n.shape
    
            n = n.flatten()
            beta = beta.flatten()            
            
            self.getAfromNB(n, beta)
            
        else:
            print "# (pyFTEGhf:init) Error: Input format not supported!"
            print "# Please specify either pair (alpha, beta) or (n, beta)"
            print

            return

        '''
        initialize array of chemical potentials
        '''
        self.mu = np.zeros((self.A.size,), dtype=np.float_)
        
        '''
        initialize array of physical observables
        '''
        self.n = np.zeros(self.mu.shape, dtype=np.float_)
        self.h = np.zeros(self.mu.shape, dtype=np.float_)
        self.s = np.zeros(self.mu.shape, dtype=np.float_)
        
        self.dmuda = np.zeros(self.mu.shape, dtype=np.float_)
        self.dmudb = np.zeros(self.mu.shape, dtype=np.float_)
        self.dnda = np.zeros(self.mu.shape, dtype=np.float_)
        self.dhda = np.zeros(self.mu.shape, dtype=np.float_)
        self.dndb = np.zeros(self.mu.shape, dtype=np.float_)
        self.dhdb = np.zeros(self.mu.shape, dtype=np.float_)

        '''
        set up differential matrices
        '''
        # (n,h) as function of (mu beta)
        self.dNdV = np.zeros((2, 2, self.mu.size,), dtype=np.float_)
        # (mu, beta) as function of (n, h)
        self.dVdN = np.zeros(self.dNdV.shape, dtype=np.float_)

        # (mu, h) as function of (n, beta)
        self.dMHdNB = np.zeros(self.dNdV.shape, dtype=np.float_)
        # (n, beta) as function of(mu, h)
        self.dNBdMH = np.zeros(self.dNdV.shape, dtype=np.float_)

        # (n, mu) as function of (beta, h)
        self.dNMdHB = np.zeros(self.dNdV.shape, dtype=np.float_)
        # (beta, h) as function of (n, mu)
        self.dHBdNM = np.zeros(self.dNdV.shape, dtype=np.float_)
        
        for i in range(self.A.size):
            # set current alpha and beta
            self.alpha = self.A[i]
            self.beta = self.B[i]
            
            converged = self.scCycle(N)
                
            if not converged:
                print "# Warning (pyFTEGhf:init): Self-consistent cylce not converged!"
                
            mu, dmuda, dmudb = self.getPotential()
            n, dnda, dndb = self.getDensity()
            h, dhda, dhdb = self.getEnergy()
            s = self.getEntropy()
            
            # compute partial derivatives of relative dimensionless chemical potential
            dadmu = 1. / dmuda
            dadb = - dmudb / dmuda
            
            # store results
            self.mu[i] = mu
            
            self.n[i] = n
            self.h[i] = h
            self.s[i] = s
        
            self.dmuda[i] = dmuda
            self.dmudb[i] = dmudb
                      
            self.dnda[i] = dnda
            self.dhda[i] = dhda
            self.dndb[i] = dndb
            self.dhdb[i] = dhdb

            '''
            Construct differential matrix of density and energy density
            as function of the physical chemical potential and the
            inverse temperature.
            '''

            self.dNdV[0,0,i] = dnda * dadmu
            self.dNdV[0,1,i] = dndb + dnda * dadb
            self.dNdV[1,0,i] = dhda * dadmu
            self.dNdV[1,1,i] = dhdb + dhda * dadb
            
            '''
            Construct all other five possible differential matrices.
            '''
            # (mu, beta) as function of (n, h)
            self.dVdN[0,0,i] = self.dNdV[1,1,i]
            self.dVdN[0,1,i] = -self.dNdV[0,1,i]
            self.dVdN[1,0,i] = -self.dNdV[1,0,i]
            self.dVdN[1,1,i] = self.dNdV[0,0,i]
    
            det_dNdV = self.dNdV[0,0,i] * self.dNdV[1,1,i] - self.dNdV[0,1,i] * self.dNdV[1,0,i]
            
            self.dVdN[:,:,i] /= det_dNdV
            
            # (mu, h) as function of (n, beta)
            self.dMHdNB[0,0,i] = 1.
            self.dMHdNB[0,1,i] = -self.dNdV[0,1,i]
            self.dMHdNB[1,0,i] = self.dNdV[1,0,i]
            self.dMHdNB[1,1,i] = det_dNdV
    
            self.dMHdNB[:,:,i] /= self.dNdV[0,0,i]
    
            # (n, beta) as function of (mu, h)
            self.dNBdMH[0,0,i] = det_dNdV
            self.dNBdMH[0,1,i] = self.dNdV[0,1,i]
            self.dNBdMH[1,0,i] = -self.dNdV[1,0,i]
            self.dNBdMH[1,1,i] = 1.
    
            self.dNBdMH[:,:,i] /= self.dNdV[1,1,i]
    
            # (n, mu) as function of (h, beta)
            self.dNMdHB[0,0,i] = self.dNdV[0,0,i]
            self.dNMdHB[0,1,i] = -det_dNdV
            self.dNMdHB[1,0,i] = 1.
            self.dNMdHB[1,1,i] = -self.dNdV[1,1,i]
    
            self.dNMdHB[:,:,i] /= self.dNdV[1,0,i]
    
            # (h, beta) as function of (n, mu)
            self.dHBdNM[0,0,i] = self.dNdV[1,1,i]
            self.dHBdNM[0,1,i] = -det_dNdV
            self.dHBdNM[1,0,i] = 1.
            self.dHBdNM[1,1,i] = -self.dNdV[0,0,i]
    
            self.dHBdNM[:,:,i] /= self.dNdV[0,1,i]

        # overwrite local temporary variables
        self.x = None
        self.dx = None
        self.nu = None
        self.q = None
        self.dqda = None
        self.dqdb = None
        
        # store alpha and beta
        self.alpha = self.A
        self.beta = self.B

        # compute free energy and grand potential
        self.f = self.h - self.s / self.beta
        self.w = self.f - self.mu * self.n

        # reshape to original shape
        self.beta = self.beta.reshape(shape)
        self.mu = self.mu.reshape(shape)
        self.n = self.n.reshape(shape)
        self.h = self.h.reshape(shape)

        self.alpha = self.alpha.reshape(shape)        
        self.s = self.s.reshape(shape)
        self.f = self.f.reshape(shape)
        self.w = self.w.reshape(shape)

        self.dNdV = self.dNdV.reshape((2,2,) + shape)
        self.dVdN = self.dVdN.reshape((2,2,) + shape)
        self.dMHdNB = self.dMHdNB.reshape((2,2,) + shape)
        self.dNBdMH = self.dNBdMH.reshape((2,2,) + shape)
        self.dNMdHB = self.dNMdHB.reshape((2,2,) + shape)
        self.dHBdNM = self.dHBdNM.reshape((2,2,) + shape)        

        return           

    def getAfromNB(self, n0, beta):
        '''
        determine the relative dimensionless chemical potential
        from density and inverse temperature
        
        input:
        n0   -> target density
        beta -> inverse temperature 
        
        output:
        alpha -> relative dimensionless chemical potential
        '''
        # set initial guess for chemical potential to zero temperature result
        kF = np.power(3. * np.pi * np.pi * n0, 1. / 3.)
        mu = .5 * kF * kF

        self.A = mu * beta
        self.B = beta

        # loop over all given pairs of n and beta
        for ii in range(n0.size):
            # set current alpha and beta 
            self.alpha = self.A[ii]
            self.beta = self.B[ii]
            
            converged = False
            for i in range(PARAMS.maxIterPotInv):
        
                N = 2**PARAMS.M
                converged = self.scCycle(N)
                
                if not converged:
                    print "# Warning (pyFTEGhf:getAfromNB): Self-consistent cylce not converged!"
                    
                n, dnda, dndb = self.getDensity()
            
                dn = dnda
                dalpha = (n0[ii] - n) / dn
    
                if (np.abs(n0[ii] - n) < PARAMS.relErrPotInv * max(1., np.abs(n0[ii]))):
                     
                    if PARAMS.debug:
                        print "# chemical potential determined in", i, "iterations"
                        print "# given density", n0[ii]
                        print "# given inverse temperature", self.beta
                        print "# dimensionless chemical potential", self.alpha
                        print
                        
                    converged = True                

                    self.A[ii] = self.alpha    
        
                    break
                
                self.alpha += dalpha
                    
            if not converged:
                print "# Warning: Inversion from (n, beta) -> alpha not converged"
                print "# inverse temperature:", self.beta
                print "# required density:", n0[ii]
                print "# achieved density:", n, "+/-", dn
                print "# dimensionless chemical potential:", self.alpha
                print "# maximum number of iterations:", PARAMS.maxIterPotInv
                print "# required accuray:", PARAMS.relErrPotInv
                print
        
        return

    def scCycle(self, N, DQ=PARAMS.dq):
        '''
        routine to solve the self-consistency condition that determines the 
        self-energy
        
        imput:
        
        N -> number of discretization points in the integration
        
        optional parameters:
        DQ -> convergence criterion for the wave vectors
        '''        
        # construct integration interval (using Gauss-Legendre integration)
        s1 = 1. / (np.exp(-self.alpha) + 1.)
        x, dx = POLY.legendre.leggauss(N)
        x = .5 * s1 * (x + 1.)
        dx = .5 * s1 * dx               
        
        # transform to energy space
        nu = (np.log((1. - x) / x) + self.alpha ) / self.beta
       
        # store integration mesh
        self.nu = nu
        self.x = x
        self.dx = dx
             
        # initial guess for momenta
        q = np.sqrt(2. * nu)

        sol = OPT.root(self.scFunc, q, method='lm', jac=True,
                       options={'maxiter': self.nSC, 'xtol': DQ, 'ftol': PARAMS.tiny})

        if sol.success:
            q = np.abs(sol.x)
            
        else:
            print sol.message
            q = np.abs(sol.x)

        # initialize source terms
        b = np.zeros((N, 2,), dtype = np.float_)
        b[:,0] = 1. / self.beta
        b[:,1] = -nu / self.beta
        
        f, A = self.scFunc(q)

        dq = LA.solve(A, b)
        
        self.q = q
        self.dqda = dq[:,0]
        self.dqdb = dq[:,1]        

        return sol.success

    def scFunc(self, x):
        '''
        linearized self-consistency equation
        '''
        q = np.abs(x)
        
        q1 = q[:,np.newaxis]
        q2 = q
            
        # compute arguments of the logarithm
        q1pq22 = np.power(q1 + q2, 2.)
        q1mq22 = np.power(q1 - q2, 2.)                       
                       
        # set diagonal to 1 in order to ensure the proper limit
        zeroMap1 = np.abs(q1mq22) < PARAMS.eta
        q1mq22[zeroMap1] = 1.
        q1pq22[zeroMap1] = 1.

        # compute ratio of q2 and q1
        x = q2 / q1
        x2 = x * x

        # compute logarithm              
        LN = np.log(q1pq22 / q1mq22)
        
        S = (self.dx * (q2 + .25 * q1 * (1. - x2) * LN)).sum(axis=-1) / np.pi
        S1 = -(self.dx * (x - .25 * (1. + x2) * LN)).sum(axis=-1) / np.pi
        S2 = -.5 * self.dx * x * LN / np.pi + 2. / np.pi * self.dx   

        # setup system of linear equations determining the momentum shifts
        # source
        f = .5 * q * q + self.C * S - self.nu
        
        # coefficient matrix
        df = self.C * S2
        
        # diagonal of coefficient matrix
        df += np.diag(q2 + self.C * S1)
        
        return f, df

    def getPotential(self):
        '''
        routine that computes the physcial chemical potential "mu"
        and its derivatives with respect to the relative dimensionless
        chemical potential "alpha" and the inverse temperature "beta".
        '''
        Cmu = 2. / np.pi
        
        mu = self.alpha / self.beta - Cmu * self.C * (self.dx * self.q).sum()
        dmuda = 1. / self.beta - Cmu * self.C * (self.dx * self.dqda).sum()
        dmudb = -self.alpha / (self.beta * self.beta) - Cmu * self.C * (self.dx * self.dqdb).sum()
        
        return mu, dmuda, dmudb
    
    def getDensity(self):
        '''
        routine that computes the density "n"
        and its derivatives with respect to the relative dimensionless
        chemical potential "alpha" and the inverse temperature "beta".
        '''
        Cn = 1. / (3. * np.pi * np.pi)

        n = Cn * (self.dx * np.power(self.q, 3.)).sum()
        dnda = Cn * 3. * (self.dx * self.dqda * np.power(self.q, 2.)).sum()
        dndb = Cn * 3. * (self.dx * self.dqdb * np.power(self.q, 2.)).sum()
        
        return n, dnda, dndb
        
    def getEnergy(self):
        '''
        routine that computes the energy "h"
        and its derivatives with respect to the relative dimensionless
        chemical potential "alpha" and the inverse temperature "beta".
        '''
        # kinetic energy contribution
        Ct = 1. / (10. * np.pi * np.pi)

        t = Ct * (self.dx * np.power(self.q, 5.)).sum()
        dtda = Ct * 5. * (self.dx * self.dqda * np.power(self.q, 4.)).sum()
        dtdb = Ct * 5. * (self.dx * self.dqdb * np.power(self.q, 4.)).sum()

        # interaction energy contribution
        Cw = 1. / np.power(2. * np.pi, 3.)

        # define q1, q2, dq1da, dq2da
        q1 = self.q[:,np.newaxis]
        q2 = self.q
        
        dq1da = self.dqda[:,np.newaxis]
        dq2da = self.dqda
        
        dq1db = self.dqdb[:,np.newaxis]
        dq2db = self.dqdb
        
        # combined integration weights
        dx = -self.dx[:,np.newaxis] * self.dx

        # compute arguments of the logarithm
        q1pq22 = np.power(q1 + q2, 2.)
        q1mq22 = np.power(q1 - q2, 2.)                       
                       
        # set diagonal to 1 in order to ensure the proper limit
        zeroMap1 = np.abs(q1mq22) < PARAMS.eta
        q1mq22[zeroMap1] = 1.
        q1pq22[zeroMap1] = 1.
        
        # compute logarithm              
        LN = np.log(q1pq22 / q1mq22)
        
        # compute square of q1 and q2 and their sum and difference
        q12 = q1 * q1
        q22 = q2 * q2
        
        q12pq22 = q12 + q22
        q12mq22 = q12 - q22
        
        # compute product of q1 and q2
        q1q2 = q1 * q2
        
        # compute E1 and E2
        E1 = q1 * (4. * q1q2 - q12mq22 * LN)
        E2 = q2 * (4. * q1q2 + q12mq22 * LN)
        
        # evaluate integrals for self-energy and its derivative
        w = Cw * (dx * (q1q2 * q12pq22 - .25 * np.power(q12mq22, 2.) * LN)).sum()
        dwda = Cw * (dx *(E1 * dq1da + E2 * dq2da)).sum()
        dwdb = Cw * (dx *(E1 * dq1db + E2 * dq2db)).sum()        

        return t + self.C * w, dtda + self.C * dwda, dtdb + self.C * dwdb

    def getEntropy(self):
        '''
        routine that computes the entropy "S"
        and its derivatives with respect to the relative dimensionless
        chemical potential "alpha" and the inverse temperature "beta".
        '''
        Cs = 1. / (self.beta * np.pi * np.pi)
      
        # compute logarithm contribution
        tmp = np.log(self.x) / (self.x - 1.) - np.log(1. - self.x) / self.x
                    
        # evaluate self-energy and its derivative
        S, dS = self.SdS(self.q)
        
        # compute derivative of dispersion relation
        dnu = self.q + self.C * dS
        
        s = Cs * (self.dx * np.power(self.q, 2.) / dnu * tmp).sum()
        
        return s

    def SdS(self, k):
        '''
        Returns the self-energy evaluated a wave vectors k. The evaluation is given
        in terms of the inversion of the disperion relation on the energy mesh.
    
        input:
        k  -> wave vector for which the self-energy is returned
        '''
        # expand dimensions of wave vectors
        k = k[...,np.newaxis]
        
        # asign local variables to global q and ds        
        # include prefactor into integration weight
        dx = -1. / np.pi * self.dx
        q = self.q
        
        # compute arguments of the logarithm (include infinitesimal)
        kpq2 = np.power(k + q, 2.) + PARAMS.eta
        kmq2 = np.power(k - q, 2.) + PARAMS.eta                    

        # compute logarithm              
        LN = np.log(kpq2 / kmq2)
        
        # compute ratio of q and k
        x = q / k
        x2 = x * x
        
        # evaluate integrals for self-energy and its derivative
        S = (dx * (q - .25 * k * (1. - x2) * LN)).sum(axis=-1)
        dS = (dx * (x - .25 * (1. + x2) * LN)).sum(axis=-1)
        
        return S, dS        