#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:42:37 2020

@author: anon
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
# Error functions
# Number
def err_num(w, ft, t, y, *args):
    return ft(w, t, *args) - y

def err_num_p(w, ft, t, y, p, *args):
    return np.power(ft(w, t, *args) - y, 2*p + 1)

#Angle
def _res_ang(a,b):

    sa = np.sin(a)
    ca = np.cos(a)

    sb = np.sin(b)
    cb = np.cos(b)

    sr = sa * cb - ca * sb
    cr = cb * ca + sb * sa

    return np.arctan2(sr, cr)

def err_ang(w, ft, t, y, *args):
    return _res_ang(ft(w, t, *args), y)

def err_ang_p(w, ft, t, y, p, *args):
    return np.power(_res_ang(ft(w, t, *args), y), 2*p+1)
#==============================================================================
#SGP4 MOE models
def inclot(w, t):
    return w[0] + t * w[1]

def nodeot(w, t):
    return w[0] + t * (w[1] + t * w[2])

def eccot(w, t):
    return w[0] + w[1] * t

def argpot(w, t):
    return w[0] + w[1] * t

#Mean frequency for linear estimation
def no_lin(w, t):
    n = w[9]*10
    for i,wi in reversed(list(enumerate(w[:9]))):
        n *= t
        n += wi * (i+1)
    return n

#Integrated meanfrequency for mean anomaly estimation
def no_int(w, t):
    m = w[9]
    for i,wi in reversed(list(enumerate(w[:9]))):
        m *= t
        m += wi
    return t*m*_MPD

def mot(w, t, n):
    return no_int(n, t) + w[0] - t*(w[1] + t*(w[2] + t*(w[3] + t*(w[4] + t*w[5]))))


def no_kozait(w, t):
    n = w[0] / np.power(1 - t * (w[1] + t * (w[2] + t * (w[3] + t * w[4]))), 3)
    return n

#==============================================================================

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
    
    #Estimate no_kozai with linear model
    y = moe[:,5]
    r0 = np.array([y[0], 0,0,0,0])
    r = leastsq(err_num, r0, args=(no_kozait, tt-tt[0], y), full_output=1)
    n = r[0]
    #plt.plot(tt, err_num(n, no_kozait, tt-tt[0], moe[:,5]), '.')
    print(n[0] - xs.no_kozai)
    
    y = no_kozait(n, tt-tt[0]) #Нужно использовать значения no_kozait
    # Причем, во время инференса тоже! Ибо интегрирование же!
    r0 = np.array([y[0], 0,0,0,0,0,0,0,0,0])
    r = leastsq(err_num, r0, args=(no_lin, tt-tt[0], y), full_output=1)
    nl = r[0]
    #plt.plot(tt, err_num(nl, no_lin, tt-tt[0], moe[:,5]), '.')
    print(nl[0] - xs.no_kozai)
    
    #Estimate mean anomaly
    y = moe[:,4]
    r0 = np.array([y[0], 0,0,0,0,0])
    r = leastsq(err_ang, r0, args=(mot, tt-tt[0], y, nl), full_output=1)
    m = r[0]
    #plt.plot(tt, err_ang(m, mot, tt-tt[0], moe[:,4], nl), '.')
    print(np.fmod(mot(m, xt[0] - tt[0], nl) - xs.mo, np.pi))
    
    #Estimate perigee argument
    y = moe[:,3]
    r0 = np.array([y[0], 0])
    r = leastsq(err_ang, r0, args=(argpot, tt-tt[0], y), full_output=1)
    argp = r[0]
    #plt.plot(tt, err_ang(argp, argpot, tt-tt[0], moe[:,3]), '.')
    print(np.fmod(argpot(argp, xt[0] - tt[0]) - xs.argpo, np.pi))
    
    y = moe[:,2]
    r0 = np.array([y[0], 0])
    r = leastsq(err_num, r0, args=(eccot, tt-tt[0], y), full_output=1)
    ecc = r[0]
    #plt.plot(tt, err_ang(ecc, eccot, tt-tt[0], moe[:,2]), '.')
    print(ecc[0] - xs.ecco)
    
    y = moe[:,1]
    r0 = np.array([y[0], 0,0])
    r = leastsq(err_ang, r0, args=(nodeot, tt-tt[0], y), full_output=1)
    node = r[0]
    #plt.plot(tt, err_ang(node, nodeot, tt-tt[0], moe[:,1]), '.')
    print(np.fmod(nodeot(node, xt[0] - tt[0]) - xs.nodeo, np.pi))
    
    y = moe[:,0]
    r0 = np.array([y[0], 0])
    r = leastsq(err_num, r0, args=(inclot, tt-tt[0], y), full_output=1)
    incl = r[0]
    #plt.plot(tt, err_ang(incl, inclot, tt-tt[0], moe[:,0]), '.')
    print(incl[0] - xs.inclo)
    
    yr = []
    yv = []
    ye = []
    for t in xt:
        inclo = inclot(incl, t-tt[0])
        nodeo = nodeot(node, t-tt[0])
        ecco = eccot(ecc, t-tt[0])
        argpo = argpot(argp, t-tt[0])
        mo = mot(m, t-tt[0], nl)
        no_kozai = no_kozait(n, t-tt[0])
        
        s = PySatrec.new_sat(t, 0, 0, 0, inclo, nodeo, ecco, argpo, mo, no_kozai)
        fr, jd = np.modf(t)
        e,r,v = s.sgp4(jd, fr)
        
        ye.append(e)
        yr.append(np.array(r).reshape(3,1))
        yv.append(np.array(v).reshape(3,1))
        
    yr = np.concatenate(yr, axis=1).T
    yv = np.concatenate(yv, axis=1).T
        
    plt.plot(xt-xt[0], xr-yr)
    
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
    