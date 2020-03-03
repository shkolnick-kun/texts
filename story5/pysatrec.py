#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:29:13 2020

@author: anon
"""
import julian
from sgp4.ext import jday
from sgp4.earth_gravity import wgs72
from sgp4.propagation import sgp4init
from sgp4.model import Satrec, Satellite

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
    
    satrec.epochyr    = y
    satrec.epochdays  = _epoch - yjd + 1
    satrec.jdsatepoch = _epoch
    
    satrec.ndot     = _ndot
    satrec.nddot    = _nddot
    satrec.bstar    = _bstar
    
    satrec.inclo    = _inclo
    satrec.nodeo    = _nodeo
    satrec.ecco     = _ecco
    satrec.argpo    = _argpo
    satrec.mo       = _mo
    satrec.no_kozai = _no_kozai    
    satrec.error    = 0;
  
    #  ---------------- initialize the orbit at sgp4epoch -------------------
    sgp4init(satrec.whichconst, satrec.opsmode, satrec.satnum, 
             satrec.jdsatepoch-2433281.5, satrec.bstar, satrec.ndot, 
             satrec.nddot, satrec.ecco, satrec.argpo, satrec.inclo, satrec.mo, 
             satrec.no_kozai, satrec.nodeo, satrec)

    return satrec

class PySatrec(Satrec):
    
    @classmethod
    def NewSat(cls, epoch, ndot, nddot, bstar, inclo, nodeo, ecco, argpo,
                  mo, no_kozai, satnum=0):
        self = cls()
        sat_construct(epoch, ndot, nddot, bstar, inclo, nodeo, ecco, argpo,
                  mo, no_kozai, satnum, self)
        return self

if __name__ == '__main__':
    import time
    import numpy as np
    
    #NORAD satnum: 17589
    s = PySatrec.NewSat(2458849.5, 0, 0, 1.8188e-05, 
                        1.23769, 5.64341, 0.0020959, 4.99355, 1.28757, 0.061648)
    
    fr, jd = np.modf(s.jdsatepoch + 10.3)
    
    print(jd, fr)
    print(s.sgp4(jd, fr))
    print(s.sgp4(jd, fr + .001))
    
    start = time.time()
    r = []
    for i in range(100000):
        #NORAD satnum: 40485
        s = PySatrec.NewSat(2458849.5, -5.94503e-11, 0, 0, 
                        0.425707, 2.10165, 0.877726, 0.312501, 4.96436, 0.00124624)
        fr, jd = np.modf(s.jdsatepoch + .001*i)
        r.append(s.sgp4(jd, fr)[0])
    end = time.time()
    
    print(end - start)
    print(sum(r))
    
    
