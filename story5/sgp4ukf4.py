#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:42:37 2020

@author: anon
"""
from math import fmod
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from orbital.constants import earth_mu, earth_radius_equatorial
from orbital.utilities import elements_from_state_vector, mean_anomaly_from_true
from pysatrec import PySatrec

#Default limitation values
_TTL  = 31.         # Time to live in days  
_MAXR = 152000e+3   # Apogee of the highest orbit ("Intergal")
_MINR = earth_radius_equatorial + 160e+3  # Perigee of the lowest orbit (stratosphere, LOL)
_MAXN = np.sqrt(earth_mu/_MINR**3) * 60. # Highest mean motion
_MINN = np.sqrt(earth_mu/_MAXR**3) * 60. # Lowest mean motion

#State vector indexes
_KEP_NUM = [0,2,5,6,7,8,9,10,11,12,13,14,15]
_KEP_ANG = [1,3,4]   #Angle state dimensions
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

def _norm_kep(_x):
    x = _x.copy()
    for i in _KEP_ANG:
        x[i] = fmod(x[i], 2.*np.pi)
        if x[i] < 0.:
            x[i] = 2.*np.pi - x[i]
    return x
#==============================================================================
#Get initial estimate of orbital elements
def get_kepler_elements(r,v):
    kep = elements_from_state_vector(np.array(r)*1000, np.array(v)*1000, earth_mu)
    mo = mean_anomaly_from_true(kep.e, kep.f)
    no_kozai = 60.0 * (earth_mu / (kep.a**3))**0.5
    return np.array((kep.i, kep.raan, kep.e, kep.arg_pe, mo, no_kozai))#.reshape((6,1))
#==============================================================================
def eval_tle(x, _epoch, _bstar):
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
_D2R   = np.pi/180.      # Degrees to radians
_MPD   = 1440.           # Minutes in day
_XD2RM = _MPD/(2.*np.pi) # rounds/ray. -> radians/minute

#State sdt will be machine epsilones for values in TLE
_AEPS = .5e-4 * _D2R
#                  inclo,   nodeo,  ecco, argpo,     mo,     no_kozai
_EPS_TLE = np.array([_AEPS,  _AEPS, .5e-7,  _AEPS, _AEPS, .5e-8*_XD2RM])

def _eval_dif(dx, thr):
    return np.all(np.abs(dx) < thr * _EPS_TLE)
    
    
# Подгонка модели 
def fit_sgp4_bstar_fixed(__r, __v, __epoch, __bstar=0, g0 = 1., eps=.5, stop_thr=1e-0, max_iter=1000):
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
        # Невязки
        
        y, exc = eval_tle(x, __epoch, __bstar)
        if (exc):
            break

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

    return x, j, _eval_dif(dx, stop_thr), np.abs(dx)/_EPS_TLE, exc
#==============================================================================
def get_initial_estimate(r,v):
    x = np.zeros((16,),dtype=np.float)
    x[:6] = get_kepler_elements(r,v).reshape((6,))
    return x

#Observation residual function
def _res_pe(a,b):
    s = b #Other hipoteses didn't work
    rn = np.linalg.norm(s[:3])
    vn = np.linalg.norm(s[3:])
    d = a - b
    d[:3] /= rn
    d[3:] /= vn
    return d

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

#State mean function
def _mean_sgp4(sigmas, Wm):
    x = np.zeros(16)
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

# State transition function
def sgp4_fx(x, dt, **fx_args):   
    return x

# Observation function
# hx args: epoch, current
def sgp4_hx(x, **hx_args):
    t1 = hx_args['current']
    t2 = t1*t1
    t3 = t2*t1
    t4 = t3*t1
    # inclo,   nodeo,  ecco, argpo,     mo,     no_kozai
    inclo = x[0] +  x[6] * t1 #Deep space!
    nodeo = x[1] +  x[7] * t1 +  x[8] * t2
    ecco  = x[2] -  x[9] * t1
    argpo = x[3] + x[10] * t1
    mo    = x[4] + x[11] * t1
    a     = x[5] - x[12] * t1 - x[13] * t2 - x[14] * t3 - x[15] * t4
    return np.array((inclo, nodeo, ecco, argpo, mo, a))


_D2R   = np.pi/180.      # Degrees to radians
_MPD   = 1440.           # Minutes in day
_XD2RM = _MPD/(2.*np.pi) # rounds/ray. -> radians/minute

#State sdt will be machine epsilones for values in TLE
_AEPS = .5e-4 * _D2R
#                  inclo,   nodeo,  ecco, argpo,     mo,     no_kozai
_X_STD = np.array([_AEPS,  _AEPS, .5e-7,  _AEPS, _AEPS, .5e-8*_XD2RM]+[1e-2]*10)

#Default observation std
_Z_STD = _X_STD[:6]

_DIMX = 16 # The number of dtate dimmentions
_DIMZ = 6  # The number of observation dimmentions

class KalmanSGP4Estimator(object):
    def __init__(self, r, v, t, cb=None, m=.1, Filter=None, sp=None, R=None):

        dt = t[1] - t[0]

        zz = []
        tt = []
        for i,ri in enumerate(r):

            z,j,c,d,exc = fit_sgp4_bstar_fixed(ri, v[i], t[i], max_iter=50)

            if c:
                zz.append(z.reshape((6,1)))
                tt.append(t[i] - t[0])

            if cb:
                cb(i)
                
        zz = np.concatenate(zz, axis=1).T
        zz[:, 5] = np.power(earth_mu*np.power((60 / zz[:,5]), 2.), (1. / 6.))
        self.z = zz
        self.t = tt
        self.epoch = t[0]

        if sp is None:
            sp = MerweScaledSigmaPoints(_DIMX, alpha=.001, beta=2., kappa=3-_DIMX)

        if Filter is None:
            Filter = UnscentedKalmanFilter

        kf = Filter(dim_x=_DIMX, dim_z=_DIMZ, dt=dt, fx=sgp4_fx, hx=sgp4_hx, 
                    points=sp, residual_x=_res_sgp4, x_mean_fn=_mean_sgp4, 
                    residual_z=_res_kep)

        kf.x = np.zeros(_X_STD.shape)
        kf.x[:6] = zz[0].reshape((6,))

        kf.Q = np.diag(_X_STD*_X_STD)

        if R is None:
            kf.R = np.diag(_Z_STD*_Z_STD)
        else:
            kf.R = R

        kf.P *= m

        self.kf = kf

    def run_one_epoch(self, shuffle=False, cb=None):
        ii = list(range(len(self.t)))

        if shuffle:
            ii = reversed(ii)
            #np.random.shuffle(ii)

        for j,i in enumerate(ii):
            self.kf.predict()
            self.kf.update(self.z[i], current=self.t[i])
            if cb:
                cb(self, i, j)
                
    @property
    def model(self):
        #Так bstar не оценить!!!
        #s = PySatrec.new_sat(self.t[0], 0, 0, 1e-9, *list(self.kf.x[:6]))
        
        bstar = 1e-9#np.clip(self.kf.x[8] / s.cc4, -.99999, .99999)
        
        return PySatrec.new_sat(self.epoch, 0, 0, bstar, *list(self.kf.x[:6]))



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
    est = KalmanSGP4Estimator(xr, xv, xt, cb=pb.update)
    pb.finish()

    y = []
    x = []
    def kf_cb(estimator, i, j):
        pb.update(j)
        y.append(np.linalg.norm(estimator.kf.y))
        x.append(estimator.kf.x[:6])
    
    pb = bar().start(len(est.z))
    pb.start()
    est.run_one_epoch(cb=kf_cb)
    #est.run_one_epoch(shuffle=True, cb=kf_cb)
    pb.finish()
    
    plt.plot(y)
    print(xs.inclo    - est.kf.x[0])
    print(xs.nodeo    - est.kf.x[1])
    print(xs.ecco     - est.kf.x[2])
    print(xs.argpo    - est.kf.x[3])
    print(xs.mo       - est.kf.x[4])
#    print(xs.no_kozai - est.kf.x[5])
    
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
    