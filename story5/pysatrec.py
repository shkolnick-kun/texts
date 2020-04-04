#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:29:13 2020

@author: anon

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
import julian
#from math import exp

from sgp4.ext import jday
from sgp4.earth_gravity import wgs72


from sgp4propagation import sgp4init
from sgp4model import Satrec, Satellite#, minutes_per_day

#from orbital.constants import earth_mu
#from scipy.optimize import root_scalar

def sat_construct(_epoch, _ndot, _nddot, _bstar, _inclo, _nodeo, _ecco, _argpo,
                  _mo, _no_kozai, _satnum, _sat=None, _whichconst=wgs72, _opsmode='i'):

    #Construct the satellite
    if _sat is None:
        satrec = Satellite()
    else:
        satrec = _sat

    satrec.satnum     = _satnum
    satrec.whichconst = _whichconst
    satrec.opsmode    = _opsmode

    dt  = julian.from_jd(_epoch, fmt='jd')
    y   = dt.year - 2000
    yjd = jday(dt.year, 1, 1, 0, 0, 0)

    satrec.epoch      = dt
    satrec.epochyr    = y
    satrec.epochdays  = _epoch - yjd + 1
    satrec.jdsatepoch = _epoch

    satrec.ndot     = _ndot
    satrec.nddot    = _nddot
    
    if _bstar > .99999:
        _bstar = .99999
    if _bstar < -.99999:
        _bstar = -.99999  
    satrec.bstar    = _bstar

    satrec.inclo    = _inclo
    satrec.nodeo    = _nodeo
    
    if _ecco < 1e-7:
        _ecco = 1e-7

    if _ecco > .9999999:
        _ecco = .9999999
    satrec.ecco     = _ecco
    
    satrec.argpo    = _argpo
    satrec.mo       = _mo
    
    if _no_kozai < 1e-7:
        _no_kozai = 1e-7
    satrec.no_kozai = _no_kozai    
    satrec.error    = 0;
    satrec.error_message = None

    #  ---------------- initialize the orbit at sgp4epoch -------------------
    sgp4init(satrec.whichconst, satrec.opsmode, satrec.satnum, 
             satrec.jdsatepoch-2433281.5, satrec.bstar, satrec.ndot, 
             satrec.nddot, satrec.ecco, satrec.argpo, satrec.inclo, satrec.mo, 
             satrec.no_kozai, satrec.nodeo, satrec)

    return satrec

class PySatrec(Satrec):

    @classmethod
    def new_sat(cls, epoch, ndot, nddot, bstar, inclo, nodeo, ecco, argpo,
                  mo, no_kozai, satnum=0):
        self = cls()
        sat_construct(epoch, ndot, nddot, bstar, inclo, nodeo, ecco, argpo,
                  mo, no_kozai, satnum, self)
        return self

#    def bstar_lime(self, tsince):
#        tsince *=  minutes_per_day
#
#        demax = self.cc4 * tsince
#        demin = demax
#        
#        ul =  0.99999
#        ll = -0.99999
#        
#        if self.isimp != 1:
#            demax -= self.cc5 * (1.0 + self.sinmao)
#            demin += self.cc5 * (1.0 - self.sinmao)
#        
#        re = 1 - self.ecco
#        #e0 - demax*b < 1 <=> e0 - 1 < demax*b
#        if demax > 0:
#            #b > -(1-e0)/demax
#            ll = max(ll, -re/demax)
#        elif demax < 0:
#            #b < (1-e0)/abs(demax) == -(1-e0)/demax
#            ul = min(ul, -re/demax)
#        
#        #e0 - demin*b > 0 <=> e0 > demin*b
#        if demin > 0:
#            #b < e0/demin
#            ul = min(ul, self.ecco/demin)
#        elif demin < 0:
#            #b > -e0/abs(demin) == e0/demin
#            ll = max(ll, self.ecco/demin)
#        
#        return ll, ul
#
#    def bstar_lima(self, tsince, amin):
#
#        t = tsince * minutes_per_day / self.bstar
#
#        ao  = (earth_mu / ((self.no_unkozai/60) ** 2.0)) ** (1.0/3.0)
#        thr = (amin/ao) ** 0.5
#        
#        def lim_b(b):
#            if b < 0:
#                return max(b, -0.99999)
#            else:
#                return min(b,  0.99999)
#
#        b0 = lim_b((1.0 - thr) / (self.cc1*t))
#        
#        if self.isimp == 1 or b0 == -0.99999:
#            return b0
#        else:
#            def fb(b):
#                t1 = b * t
#                t2 = t1 * t1
#                t3 = t2 * t1
#                t4 = t3 * t1
#                v  = thr - 1.
#                v += self.cc1*t1 + self.d2*t2 + self.d3*t3 + self.d4*t4
#                return v
#
#            def fbdot(b):
#                t1 = b * t
#                t2 = t1 * t1
#                t3 = t2 * t1
#                v  = self.cc1 + 2.*self.d2*t1 + 3.*self.d3*t2 + 4.*self.d4*t3
#                return v*t
#
#            r = root_scalar(fb, fprime=fbdot, x0=b0)
#            
#            if not r.converged:
#                def fbddot(b):
#                    t1 = b * t
#                    t2 = t1 * t1
#                    v  = 2.*self.d2 + 6.*self.d3*t1 + 12.*self.d4*t2
#                    return v*t*t
#                return root_scalar(fbdot, fprime=fbddot, x0=0).root
#            else:
#                return lim_b(r.root)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sgp4.model import Satrec as PySt
    
    l1 = '1 44249U 19029Q   20034.91667824  .00214009  00000-0  10093-1 0  9996'
    l2 = '2 44249  52.9973  93.0874 0006819 325.3043 224.0257 15.18043020  1798'
    xs = PySt.twoline2rv(l1, l2)
    
    delta = float(2. * np.pi / (xs.no_kozai * 1440.))/50 #50 точек на период
    epoch = xs.jdsatepoch      #Начало эпохи
    fr, jd = np.modf(np.array([epoch + delta*k for k in range(int(31./delta)+ 1)]))
    
    xe,xr,xv = xs.sgp4_array(jd, fr)
    
    ys = PySatrec.new_sat(xs.jdsatepoch, xs.ndot, xs.nddot, xs.bstar, xs.inclo,
                         xs.nodeo, xs.ecco, xs.argpo, xs.mo, xs.no_kozai)
    ye,yr,yv = xs.sgp4_array(jd, fr)
    
    plt.plot(xr-yr)
    
