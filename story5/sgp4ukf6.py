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
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Saver
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


_DIMX = 12 # The number of dtate dimmentions
_DIMZ = 6  # The number of observation dimmentions
#                  inclo,   nodeo,  ecco, argpo,     mo,     no_kozai
_X_STD = np.array([1e-5]*_DIMX)
_X_STD[:6] = _EPS_TLE
#_X_STD *= 1e-1

#Default observation std
_Z_STD = _EPS_TLE

#State vector indexes
#Angle state dimensions
_KEP_ANG = [1,3,4]
#Number like state dimensions
_KEP_NUM = [i for i in range(_DIMX) if i not in _KEP_ANG]
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
#==============================================================================
def _norm_kep(_x):
    x = _x.copy()
    for i in _KEP_ANG:
        x[i] = fmod(x[i], 2.*np.pi)
        if x[i] < 0.:
            x[i] = 2.*np.pi - x[i]
    return x
#==============================================================================
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
#==============================================================================
def _eval_dif(dx, thr):
    return np.all(np.abs(dx) < thr * _EPS_TLE)
#==============================================================================    
# Computes SGP4 Mean Orbital Elements from state (r,v) at epoch
# See https://apps.dtic.mil/dtic/tr/fulltext/u2/a289281.pdf
def sgp4_moe_from_state(__r, __v, __epoch, __bstar=0, g0 = 1., eps=.5, 
                         stop_thr=1e-0, max_iter=1000):
    #Наблюдаемые значения
    yo = get_kepler_elements(__r,__v)

    #Начальное приближение
    x = yo.copy()

    # Коэффициенты усиления ошибок
    gains = np.ones(x.shape, dtype=np.float) * g0

    # Информация о знаках невязок
    sez = yo >= x

    exc = 0

    for j in range(max_iter):
        y, exc = _eval_sat(x, __epoch, __bstar)
        if (exc):
            break
        
        # Невязки
        e = _res_kep(yo, y)

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
#==============================================================================
# Computes SGP4 Mean Orbital Elements from arrays r,v,t
def sgp4_moe_from_arrays(r, v, t, cb=None):
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
#==============================================================================
#State residual function
def _res_sgp4(a,b):
    r = a - b

    sa = np.sin(a[_KEP_ANG])
    ca = np.cos(a[_KEP_ANG])

    sb = np.sin(b[_KEP_ANG])
    cb = np.cos(b[_KEP_ANG])

    sr = sa * cb - ca * sb
    cr = cb * ca + sb * sa

    r[_KEP_ANG] = np.arctan2(sr, cr)

    return r
#==============================================================================
#State mean function
def _mean_sgp4(sigmas, Wm):
    x = np.zeros(_DIMX)
    #Sin and Cos of angles
    si = np.zeros(3)
    co = np.zeros(3)
    for i, s in enumerate(sigmas):
        x[_KEP_NUM] += s[_KEP_NUM]*Wm[i]
        si += np.sin(s[_KEP_ANG])*Wm[i]
        co += np.cos(s[_KEP_ANG])*Wm[i]
    #Compute mean amgles
    ang = np.arctan2(si, co)
    #Angle normalization
    for i in range(3):
        if ang[i] < 0.:
            ang[i] += 2*np.pi
    #Fill angles
    x[_KEP_ANG] = ang

    return x
#==============================================================================
# State transition function
def _sgp4_fx(x, dt, **fx_args):   
    x[0] += dt * (x[6])                         #inclo
    x[1] += dt * (x[7])                         #nodeo
    x[2] += dt * (x[8])                         #ecco
    x[3] += dt * (x[9])                         #argpo
    x[4] += dt * (x[10] + x[5] + dt * x[11])    #mo
    x[5] += dt * (x[11] * 2)                    #no_kozai
    return x
#==============================================================================
# Observation function
# hx args: epoch, current
def _sgp4_hx(x, **hx_args):
    return x[:6]
#==============================================================================
class KalmanSGP4MOEEstimator(object):
    def __init__(self, cb=None, m=.1, Filter=None, sp=None, R=None):
        self.epoch = epoch

        if sp is None:
            sp = MerweScaledSigmaPoints(_DIMX, alpha=.001, beta=2., kappa=3-_DIMX)

        if Filter is None:
            Filter = UnscentedKalmanFilter

        kf = Filter(dim_x=_DIMX, dim_z=_DIMZ, dt=0, fx=_sgp4_fx, hx=_sgp4_hx, 
                    points=sp, residual_x=_res_sgp4, x_mean_fn=_mean_sgp4, 
                    residual_z=_res_kep)

        kf.Q = np.diag(_X_STD*_X_STD)

        if R is None:
            kf.R = np.diag(_Z_STD*_Z_STD)
        else:
            kf.R = R

        kf.P *= m

        self.kf = kf

    def run(self, t, moe, cb=None, shuffle=False):
        saver = Saver(self.kf)
        self.kf.x[:6] = moe[0]
        self.kf.z     = moe[0]
        self.kf._dt   = 0
        saver.save()

        for i,zi in enumerate(moe[1:]):
            self.kf.predict(dt = t[i+1] - t[i])
            self.kf.update(zi)
            saver.save()
            if cb:
                cb(self, i)
                
        saver.to_array()
        x, P, K = self.kf.rts_smoother(saver.x, saver.P, dts=saver._dt)
        
        return x
                
#    def predict(self, t):
#        return _moe_estimate(self.kf.x, t-self.epoch)
#                
#    @property
#    def model(self):
#        #Так bstar не оценить!!!
#        #s = PySatrec.new_sat(self.t[0], 0, 0, 1e-9, *list(self.kf.x[:6]))
#        
#        bstar = 1e-9#np.clip(self.kf.x[8] / s.cc4, -.99999, .99999)
#        
#        return PySatrec.new_sat(self.epoch, 0, 0, bstar, *list(self.kf.x[:6]))

#==============================================================================
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

    

    pb = bar().start(len(xt))
    pb.start()
    moe, tt = sgp4_moe_from_arrays(xr,xv,xt,cb=pb.update)
    pb.finish()
    
    #Приводим к суткам, чтобы размерность всех скоростей в фильтре совпадала
    moe[:,5] *= _MPD

    est = KalmanSGP4MOEEstimator()
    y = []
    x = []
    def kf_cb(estimator, j):
        pb.update(j)
        y.append(estimator.kf.y.reshape(6,1))
        x.append(estimator.kf.x[:6].reshape(6,1))
    
    pb = bar().start(len(moe))
    pb.start()
    sx=est.run(tt, moe, cb=kf_cb, shuffle=False)
    pb.finish()
    
    y = np.concatenate(y, axis=1).T
    x = np.concatenate(x, axis=1).T
    plt.plot(y[10:])
    print(xs.inclo         - sx[0,0])
    print(xs.nodeo         - sx[0,1])
    print(xs.ecco          - sx[0,2])
    print(xs.argpo         - sx[0,3])
    print(xs.mo            - sx[0,4])
    print(xs.no_kozai*_MPD - sx[0,5])
    
    #ys = est.model
    #ye,yr,yv = ys.sgp4_array(jd, fr)
    
    #print(xs.inclo    - ys.inclo)
    #print(xs.nodeo    - ys.nodeo)
    #print(xs.ecco     - ys.ecco)
    #print(xs.argpo    - ys.argpo)
    #print(xs.mo       - ys.mo)
    #print(xs.no_kozai - ys.no_kozai)
    #print(ys.bstar)
    
    #plt.plot(xt, xr - yr)
    