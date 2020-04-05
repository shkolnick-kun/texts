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
    
    _MAXB = .99999
    if _bstar > _MAXB:
        _bstar = _MAXB
    if _bstar < -_MAXB:
        _bstar = -_MAXB
    satrec.bstar    = _bstar

    satrec.inclo    = _inclo
    satrec.nodeo    = _nodeo
    
    _MINE = .5e-7
    if _ecco < _MINE:
        _ecco = _MINE
        
    _MAXE = 1. - _MINE
    if _ecco > _MAXE:
        _ecco = _MAXE
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
    
