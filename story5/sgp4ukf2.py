#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:42:37 2020

@author: anon
"""
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Saver
from orbital.constants import earth_mu, earth_radius_equatorial
from orbital.utilities import elements_from_state_vector, mean_anomaly_from_true
from pysatrec import PySatrec

#Default limitation values
_TTL  = 31.         # Time to live in days  
_MAXR = 152000e+3   # Apogee of the highest orbit ("Intergal")
_MINR = earth_radius_equatorial + 50e+3  # Perigee of the lowest orbit (stratosphere, LOL)
_MAXN = np.sqrt(earth_mu/_MINR**3) * 60. # Highest mean motion
_MINN = np.sqrt(earth_mu/_MAXR**3) * 60. # Lowest mean motion

_D2R   = np.pi/180.      # Degrees to radians
_MPD   = 1440.           # Minutes in day
_XD2RM = _MPD/(2.*np.pi) # rounds/ray. -> radians/minute

#Other consants
_KEP_ANG = [1,3,4]

_DIMX = 13 # The number of dtate dimmentions
_DIMZ = 6  # The number of observation dimmentions

#State sdt will be machine epsilones for values in TLE
_AEPS = .5e-4 * _D2R
#                  inclo,   nodeo,  ecco, argpo,    mo,     no_kozai
_X_STD = np.array([_AEPS,  _AEPS, .5e-7,  _AEPS, _AEPS, .5e-8*_XD2RM]+[.5e-5]*7)

#Default observation std
_Z_STD = .00001

#------------------------------------------------------------------------------
def norm_moe(x):
    """
    Normalize orbiatl elements
    """
    y = x.copy()
    y[_KEP_ANG]  = np.fmod(y[_KEP_ANG], 2.*np.pi)
    y[_KEP_ANG] += 2 * np.pi * (y[_KEP_ANG] < 0).astype(np.float)    
#    for i in _KEP_ANG:
#        y[i] = fmod(y[i], 2.*np.pi)
#        if y[i] < 0.:
#            y[i] = 2.*np.pi - y[i]
    return y

#------------------------------------------------------------------------------
#Observation residual function
def _res_pe(a,b):
    s = b #Other hipoteses didn't work
    rn = np.linalg.norm(s[:3])
    vn = np.linalg.norm(s[3:])
    d = a - b
    d[:3] /= rn
    d[3:] /= vn
    return d

#------------------------------------------------------------------------------
# State transition function
def sgp4_fx(x, dt, **fx_args):
    
    x[4]  += dt * (x[5] + dt * 0.5 * (x[11] + x[12]))
    x[:6] += dt * x[6:12]
    
    #Eccentricity must not fall too low!
    if x[2] < _X_STD[2]:
        x[2] = _X_STD[2]
    
    return x

#------------------------------------------------------------------------------
# Observation function
# hx args: epoch, current
def sgp4_hx(x, **hx_args):
    #Make the satellite
    xn = norm_moe(x[:6])
    xn[5] /= _MPD
    hsat = PySatrec.new_sat(hx_args['epoch'], 0, 0, 0, *list(xn))
    
    #Propagate the satellite state
    hfr, hjd = np.modf(hx_args['epoch'])
    he,hr,hv = hsat.sgp4(hjd, hfr)
    
    if he > 0:
        print(x)
        print(hsat.error_message)
    
    return np.array(list(hr) + list(hv))

#------------------------------------------------------------------------------
class Sgp4MoeEstimator(object):
    def __init__(self, epoch=None, x0=None, m=.1, Filter=None, sp=None, R=None):
        
        if sp is None:
            sp = MerweScaledSigmaPoints(_DIMX, alpha=.01, beta=2., kappa=3-_DIMX)

        if Filter is None:
            Filter = UnscentedKalmanFilter
            
        kf = Filter(dim_x=_DIMX, dim_z=_DIMZ, dt=1, fx=sgp4_fx, hx=sgp4_hx, 
                    points=sp, residual_z=_res_pe)

        self.epoch = epoch
        kf.x = x0

        kf.Q = np.diag(_X_STD*_X_STD)

        if R is None:
            kf.R = np.diag([_Z_STD**2]*_DIMZ)
        else:
            kf.R = R

        kf.P *= m

        self.kf = kf
        
    def init_x(self, z):
        kep = elements_from_state_vector(z[:3]*1000, z[3:]*1000, earth_mu)
        mo = mean_anomaly_from_true(kep.e, kep.f)
        no_kozai = 60.0 * (earth_mu / (kep.a**3))**0.5
        return np.array([kep.i, kep.raan, kep.e, kep.arg_pe, mo, no_kozai*_MPD]+[0.]*7)

    def estimate(self, t, z, cb=None):
        
        dt = t[1:] - t[:-1]
        if np.any(dt < 0):
            raise ValueError('Times must be sorted in accedning order.')
        
        t = t.copy()
        z = z.copy()
            
        if self.kf.x is None:
            self.kf.x = self.init_x(z[0])
        #
        xe = []
        for i,ti in enumerate(t[1:]):
            #Prepare filtered MOE estimates
            x = self.kf.x[:6].copy()
            x[5] /= _MPD
            xe.append(x.reshape(6,1))
            
            self.kf.predict(dt=dt[i])
            self.kf.update(z[i+1], epoch=ti)
            #Partial P reset due to ecco limitation
            if self.kf.P[2,2] <= 0:
                self.kf.P[:,2] = 0
                self.kf.P[2,:] = 0
                self.kf.P[2,2] = _X_STD[2]**2
                
            #Callback
            if cb:
                cb(self, i)
        #
        x = self.kf.x[:6].copy()
        x[5] /= _MPD
        xe.append(x.reshape(6,1))
        #
        return np.concatenate(xe, axis=1).T
                
    def smooth(self, t, z):
        
        dt = t[1:] - t[:-1]
        if np.any(dt < 0):
            raise ValueError('Times must be sorted in accedning order.')
        
        t = t.copy()
        z = z.copy()
            
        if self.kf.x is None:
            self.kf.x = self.init_x(z[0])
            
        s = Saver(self.kf)
        #s.save()
        
        for i,ti in enumerate(t[1:]):
            self.kf.predict(dt=dt[i])
            self.kf.update(z[i+1], epoch=ti)
            s.save()
            
        s.to_array()
        xs,_,_ = self.kf.rts_smoother(s.x, s.P)
        return xs[:,:6]
        
            

#------------------------------------------------------------------------------
def state_from_sgp4_moe(mt, moe):
    """
    Compute state vectors from MOE
    """
    rr = []
    vv = []
    ee = []
    #
    for i,t in enumerate(mt):
        s = PySatrec.new_sat(t, 0, 0, 0, *list(moe[i].reshape(6)))
        fr, jd = np.modf(t)
        e,r,v = s.sgp4(jd, fr)
        #
        ee.append(e)
        rr.append(np.array(r).reshape(3,1))
        vv.append(np.array(v).reshape(3,1))
    #
    rr = np.concatenate(rr, axis=1).T
    vv = np.concatenate(vv, axis=1).T
    #
    return ee, rr, vv

#------------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from progressbar import ProgressBar as bar
    from sgp4.model import Satrec
    from sgp4.api import jday
    from sgp4.ext import days2mdhms
    
    def get_sat_epoch(y, d):
        y += 2000
        return sum(jday(y, *days2mdhms(y, d)))
    
    #Generate some data
    #The Hardest one!
    l1 = '1 44249U 19029Q   20034.91667824  .00214009  00000-0  10093-1 0  9996'
    l2 = '2 44249  52.9973  93.0874 0006819 325.3043 224.0257 15.18043020  1798'
    #MIN(no_kozai)
    #l1 = '1 40485U 15011D   20034.87500000 -.00001962  00000-0  00000+0 0  9996'
    #l2 = '2 40485  24.3912 120.4159 8777261  17.9050 284.4369  0.28561606 10816'
    #MIN(bstar)
    #l1 = '1 81358U          20028.49779613 -.00005615  00000-0 -72071+0 0  9998'
    #l2 = '2 81358  62.6434  61.1979 0370276 129.5311 233.8804  9.81670356    16'
    #MAX(no_kozai)
    #l1 = '1 44216U 19006CS  20035.07310469  .00413944  15423-5  43386-3 0  9995'
    #l2 = '2 44216  95.9131 264.4538 0065601 211.8276 147.5518 16.05814498 43974'
    xs = Satrec.twoline2rv(l1, l2)
    
    delta = float(2. * np.pi / (xs.no_kozai * 1440.))/50 #50 points per round
    epoch = get_sat_epoch(xs.epochyr, xs.epochdays)      #Start of epoch
    xt = np.array([epoch + delta*k for k in range(int(31./delta)+ 1)])
    fr, jd = np.modf(xt)
    xe,xr,xv = xs.sgp4_array(jd, fr)
    print(np.unique(xe))
    
    START = 0
    END   = len(xt)
    
    est = Sgp4MoeEstimator()
    zz = np.concatenate((xr[START:END],xv[START:END]), axis=1)
    tt = xt[START:END]
    
    y = []
    def kf_cb(estimator, i):
        global y
        pb.update(i)
        y.append(np.linalg.norm(estimator.kf.y))

    pb = bar().start(len(tt))
    pb.start()
    moe = est.estimate(tt, zz, cb=kf_cb)
    pb.finish()

    ye,yr,yv = state_from_sgp4_moe(tt, moe)
    print(np.unique(ye))
    
    plt.plot(xr[5:]-yr[5:])
    