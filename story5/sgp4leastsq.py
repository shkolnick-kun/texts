#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:42:37 2020

@author: anon

This is licensed under an MIT license. See the readme.MD file
for more information.
"""
from math import fmod
import numpy as np
from scipy.optimize import leastsq
#from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from orbital.constants import earth_mu, earth_radius_equatorial
from orbital.utilities import elements_from_state_vector, mean_anomaly_from_true
from pysatrec import PySatrec


#Default limitation values
_TTL  = 31.         # Time to live in days  
_MAXR = 152000e+3   # Apogee of the highest orbit ("Intergal")
_MINR = earth_radius_equatorial + 160e+3  # Perigee of the lowest orbit (stratosphere, LOL)
_MAXN = np.sqrt(earth_mu/_MINR**3) * 60. # Highest mean motion
_MINN = np.sqrt(earth_mu/_MAXR**3) * 60. # Lowest mean motion

_D2R   = np.pi/180.      # Degrees to radians
_MPD   = 1440.           # Minutes in day
_XD2RM = _MPD/(2.*np.pi) # rounds/ray. -> radians/minute

#State sdt will be machine epsilones for values in TLE
_AEPS = .5e-4 * _D2R
#                  inclo,   nodeo,  ecco, argpo,     mo,     no_kozai
_EPS_TLE = np.array([_AEPS,  _AEPS, .5e-7,  _AEPS, _AEPS, .5e-8*_XD2RM])

#State vector indexes
#Angle state dimensions
_KEP_ANG = [1,3,4]
#Number like state dimensions
#_KEP_NUM = [i for i in range(_DIMX) if i not in _KEP_ANG]
#==============================================================================
#Get initial estimate of orbital elements
def get_kepler_elements(r,v):
    kep = elements_from_state_vector(np.array(r)*1000, np.array(v)*1000, earth_mu)
    mo = mean_anomaly_from_true(kep.e, kep.f)
    no_kozai = 60.0 * (earth_mu / (kep.a**3))**0.5
    return np.array((kep.i, kep.raan, kep.e, kep.arg_pe, mo, no_kozai))

#==============================================================================
#Observation residual function
def _res_kep(a,b):
    r = a - b
    
    sa = np.sin(a[_KEP_ANG])
    ca = np.cos(a[_KEP_ANG])
    
    sb = np.sin(b[_KEP_ANG])
    cb = np.cos(b[_KEP_ANG])
    
    sr = sa * cb - ca * sb
    cr = cb * ca + sb * sa
    
    r[_KEP_ANG] = np.arctan2(sr, cr)
    
    return r
#------------------------------------------------------------------------------
def _norm_kep(_x):
    x = _x.copy()
    for i in _KEP_ANG:
        x[i] = fmod(x[i], 2.*np.pi)
        if x[i] < 0.:
            x[i] = 2.*np.pi - x[i]
    return x
#------------------------------------------------------------------------------
def _eval_sat(x, _epoch, _bstar):
    try:
        # Генерируем спутник
        ysat = PySatrec.new_sat(_epoch, 0, 0, _bstar, *list(x))
    
        # Моделируем его сотояние на момент "Эпохи"
        yfr, yjd = np.modf(_epoch)
        ye,yr,yv = ysat.sgp4(yjd, yfr)
    except Exception as ex:
        print(ex)
        ye = 5
    
    if ye != 0:
        return np.zeros(x.shape, dtype=np.float), ye
    
    # Возвращаем наблюдаемые значения элементов движения
    return get_kepler_elements(yr, yv), 0
#------------------------------------------------------------------------------
def _eval_dif(dx, thr):
    return np.all(np.abs(dx) < thr * _EPS_TLE)
#------------------------------------------------------------------------------
# Computes SGP4 Mean Orbital Elements from state (r,v) at epoch
# See https://apps.dtic.mil/dtic/tr/fulltext/u2/a289281.pdf
def sgp4_moe_from_kep(kep, __epoch, __bstar=0, g0 = 1., eps=.5, 
                         stop_thr=1e-0, max_iter=1000):

    #Начальное приближение
    x = kep.copy()

    # Коэффициенты усиления ошибок
    gains = np.ones(x.shape, dtype=np.float) * g0

    # Информация о знаках невязок
    sez = kep >= x

    exc = 0

    for j in range(max_iter):
        y, exc = _eval_sat(x, __epoch, __bstar)
        if (exc):
            break

        # Невязки
        e = _res_kep(kep, y)

        #Знаки невязок
        se = e > 0.
        changed = np.logical_xor(se, sez)
        sez = se

        # Если знак не менялся - увеличиваем коэффициент, 
        # если менялся - уменьшаем
        gains *= (1. + eps)*np.logical_not(changed) + (1. - eps)*changed
        gains = np.clip(gains, 0., 1.)

        dx = gains * e
        # Ограничения:
        # inclo[0, pi] ecco [0;.9999999], no_kozai [0; _MAXN]
        for i,ul in ((0, np.pi),(2, .9999999),(5, _MAXN)):

            if dx[i] == 0:
                continue

            k = 1

            if x[i] + dx[i] >= ul:
                k = dx[i]/(ul - x[i])/(1. - eps)

            if x[i] + dx[i] <= 0:
                k = abs(dx[i]/x[i])/(1. - eps)

            gains[i] /= k
            dx[i]    /= k

        x += dx

        x = _norm_kep(x)

        if _eval_dif(dx, stop_thr):
            break

    # Нормализация углов
    x = _norm_kep(x)

    return x, j, _eval_dif(dx, stop_thr), exc
#------------------------------------------------------------------------------

def sgp4_moe_from_state(__r, __v, __epoch, __bstar=0, g0 = 1., eps=.5, 
                         stop_thr=1e-0, max_iter=1000):
    #Наблюдаемые значения
    _kep = get_kepler_elements(__r,__v)
    
    return sgp4_moe_from_kep(_kep, __epoch, __bstar, g0, eps, stop_thr, max_iter)    
#==============================================================================
# Computes SGP4 Mean Orbital Elements from arrays r,v,t
def sgp4_moe_from_state_arrays(r, v, t, cb=None):

    xx = []
    tt = []

    for i, ri in enumerate(r):
        x,j,c,e = sgp4_moe_from_state(ri, v[i], t[i], max_iter=50)

        if c:
            xx.append(x.reshape(6,1))
            tt.append(t[i])

        if cb:
            cb(i)

    return np.concatenate(xx, axis=1).T, np.array(tt)

#------------------------------------------------------------------------------
# Computes SGP4 Mean Orbital Elements from arrays r,v,t
def sgp4_moe_from_kep_arrays(kep, t, cb=None):

    xx = []
    tt = []

    for i, ki in enumerate(kep):
        x,j,c,e = sgp4_moe_from_kep(ki, t[i], max_iter=50)

        if c:
            xx.append(x.reshape(6,1))
            tt.append(t[i])

        if cb:
            cb(i)

    return np.concatenate(xx, axis=1).T, np.array(tt)

#==============================================================================
# Error functions
# Number
def err_num(w, ft, t, y, p, *args):

    if 0 == p:
        return ft(w, t, *args) - y

    return np.power(ft(w, t, *args) - y, 2*p + 1)

#------------------------------------------------------------------------------
#Angle
def _res_ang(a,b):

    sa = np.sin(a)
    ca = np.cos(a)

    sb = np.sin(b)
    cb = np.cos(b)

    sr = sa * cb - ca * sb
    cr = cb * ca + sb * sa

    return np.arctan2(sr, cr)

#State mean function
def _mean_ang(x):
    #Sin and Cos of angles
    sm = np.mean(np.sin(x))
    cm = np.mean(np.cos(x))
    #Compute mean amgles
    return np.arctan2(sm, cm)

#State mean function
def _mean_ang_2(x):   
    #Compute mean amgles
    ang = _mean_ang(x)
    ang += 2 * np.pi * (ang < 0).astype(np.float)
    return ang

#------------------------------------------------------------------------------
def err_ang(w, ft, t, y, p, *args):

    if 0 == p:
        return _res_ang(ft(w, t, *args), y)

    return np.power(_res_ang(ft(w, t, *args), y), 2*p+1)

#==============================================================================
#SGP4 MOE models
#inclo, ecco, argpo
def flin(w, t):
    return w[0] + t * w[1]

#------------------------------------------------------------------------------
#nodeo
def fsqr(w, t):
    return w[0] + t * (w[1] + t * w[2])

#------------------------------------------------------------------------------
#no_kozai, mean motion
def no_kozait(w, t):
    n = w[0] / np.power(1 - t * (w[1] + t * (w[2] + t * (w[3] + t * w[4]))), 3)
    return n

#------------------------------------------------------------------------------
# mo = n0*t + err_mo(w,t)
    
# error model speed 
# w_verrmo = w_errmo[1:]
def verr_mot(w, t):
    e = w[4]*5
    for i,wi in reversed(list(enumerate(w[:4]))):
        e *= t
        e += wi * (i+1)
    return e

#mo error model
def err_mot(w, t):
    e = w[4]
    for i,wi in reversed(list(enumerate(w[:4]))):
        e *= t
        e += wi
    return e

#==============================================================================
def _leastsq_wr(erf, w0, args, name):

    if type(w0) is int:
        w0 = np.array([args[2][0]] + [0.] * (w0 - 1))    

    r = leastsq(erf, w0, args, full_output=1)
    if r[4] not in (1,2,3,4):
        print("Warning: %s fit did not converge!"%name)
        print(r)
    return r[0]

#==============================================================================
class SGP4MOERegression(object):
    def __init__(self, pnum=1000, p=0):
        self.wn    = None
        self.wm    = None
        self.wargp = None
        self.wecc  = None
        self.wnode = None
        self.wincl = None
        self.epoch = None
        self.pnum = pnum
        self.p = p
        
    def fit(self, t, m):
        
        p = self.p
        
        self.epoch = t[0]
        t = t.copy();
        t -= self.epoch
        
        dt = t[1:] - t[:-1]
        
        #Fit no_kozai sgp4 model
        self.wn = _leastsq_wr(err_num, 5, (no_kozait, t, m[:,5], p), "no_kozai")
              
        #mo error model
        n0 = self.wn[0]*_MPD
        err_mo = _res_ang(m[:,4], np.array([n0*ti for ti in t])) #mo error
        v = _res_ang(err_mo[1:], err_mo[:-1])/dt #mo error speed
        # Fit mo error speed first to get a good first guess
        w0     = np.array([err_mo[0],0,0,0,0,0])
        w0[1:]  = _leastsq_wr(err_ang, w0[1:], (verr_mot, t[1:], v, p), "verr_mot")   
        self.wm = _leastsq_wr(err_ang, w0,     (err_mot, t, err_mo, p), "err_mot")

        #Fit argpo model
        argpo = m[:,3]
        w0 = np.array([argpo[0], _mean_ang(_res_ang(argpo[1:], argpo[:-1])/dt)])
        self.wargp = _leastsq_wr(err_ang, w0, (flin, t, argpo, p), "argpo")
        
        #Fit ecco model
        self.wecc  = _leastsq_wr(err_num, 2, (flin, t, m[:,2], p), "ecco")
        
        #Fit nodeo model
        nodeo = m[:,1].copy()
        v  = _res_ang(nodeo[1:], nodeo[:-1])/dt
        #Fit nodeo speed first to get good initial guess for nopeo
        w0 = np.array([nodeo[0],0,0])
        w0[1:]     = _leastsq_wr(err_ang, w0[1:], (flin, t[1:], v, p), "vnod")
        w0[2]     /= 2
        self.wnode = _leastsq_wr(err_ang, w0, (fsqr, t, nodeo, p), "nodeo")

        #Fit ecco model
        self.wincl = _leastsq_wr(err_num, 2, (flin, t, m[:,0], p), "inclo")
        
    def predict(self, t):
        t = t.copy().reshape(len(t), 1)
        t -= self.epoch
        
        no_kozai = no_kozait(self.wn, t)
        mo = self.wn[0] * _MPD * t + err_mot(self.wm, t)
        argpo = flin(self.wargp, t)
        ecco  = flin(self.wecc,  t)
        nodeo = fsqr(self.wnode, t)
        inclo = flin(self.wincl, t)
        
        return np.concatenate((inclo, nodeo, ecco, argpo, mo, no_kozai), axis=1)

#==============================================================================
def state_from_sgp4_moe(mt, moe):
    yr = []
    yv = []
    ye = []

    for i,t in enumerate(mt):
        s = PySatrec.new_sat(t, 0, 0, 0, *list(moe[i]))
        fr, jd = np.modf(t)
        e,r,v = s.sgp4(jd, fr)

        ye.append(e)
        yr.append(np.array(r).reshape(3,1))
        yv.append(np.array(v).reshape(3,1))

    yr = np.concatenate(yr, axis=1).T
    yv = np.concatenate(yv, axis=1).T

    return ye, yr, yv

#==============================================================================
def get_sat_epoch(y, d):
    y += 2000
    return sum(jday(y, *days2mdhms(y, d)))
    
#==============================================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from progressbar import ProgressBar as bar
    from sgp4.model import Satrec
    from sgp4.api import jday
    from sgp4.ext import days2mdhms
    
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
    n = int(31./delta)
    xt = np.array([epoch + delta * k for k in range(n + 1)])
    fr, jd = np.modf(xt)
    xe,xr,xv = xs.sgp4_array(jd, fr)

    pb = bar().start(len(xt))
    pb.start()
    moe, tt = sgp4_moe_from_state_arrays(xr,xv,xt,cb=pb.update)
    pb.finish()
    
    est = SGP4MOERegression(p=0)
    est.fit(tt, moe)
    em = est.predict(tt)
    plt.plot(tt-tt[0], _res_kep(moe.T,em.T).T)
       
    em = est.predict(xt)    
    ye,yr,yv = state_from_sgp4_moe(xt, em)
        
    #plt.plot(xt-xt[0], xr-yr)
    