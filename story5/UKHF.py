# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from copy import deepcopy
import numpy as np
from numpy import eye, dot, isscalar

from scipy.stats import chi2
from filterpy.kalman import unscented_transform
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import pretty_str


class UKHF(UnscentedKalmanFilter):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    r"""
    Implements Unscented Kalman-Hinfinity Filter 
    See UnscentedKalmanFilter.
    """

    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None, residual_z=None, alpha=0.01):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """
        #pylint: disable=too-many-arguments
        UnscentedKalmanFilter.__init__(self, dim_x, dim_z, dt, hx, fx, points,
                                       sqrt_fn=sqrt_fn,
                                       x_mean_fn=x_mean_fn, z_mean_fn=z_mean_fn,
                                       residual_x=residual_x, residual_z=residual_z)
        
        self.beta_n = chi2.ppf(1.0 - alpha, dim_z)

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)
        
        # mean and covariance of prediction passed through unscented transform
        zp, C = UT(self.sigmas_h, self.Wm, self.Wc, None, self.z_mean, self.residual_z)
        self.S = C + R
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)
                    
        nu = self.residual_z(z, zp)   # residual
        thr = self.beta_n
        if dot(nu.T, dot(self.SI, nu)) > thr: 
            #Divergence detected, H-infinity correction needed
            nutnu = dot(nu.T, nu)
            k  = nutnu * nutnu / thr - dot(nu.T, dot(self.S, nu))
            k /= dot(nu.T, dot(C , nu))

            #Update self.P
            m = 1.0 + k
            self.P *= m

            #Update self.S, self.SI, Pxz due to self.P update
            #In linear Kalman filter Pxy = PHT, S = HPHT+R, if P *= m, then 
            Pxz *= m 
            self.S = m*C + R
            self.SI = self.inv(self.S)
            
        self.y = nu
        self.K = dot(Pxz, self.SI)        # Kalman gain
        
        # update Gaussian state estimate (x, P)
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def __repr__(self):
        return '\n'.join([
            'UnscentedKalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('S', self.S),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('likelihood', self.likelihood),
            pretty_str('mahalanobis', self.mahalanobis),
            pretty_str('sigmas_f', self.sigmas_f),
            pretty_str('h', self.sigmas_h),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('residual_x', self.residual_x),
            pretty_str('residual_z', self.residual_z),
            pretty_str('msqrt', self.msqrt),
            pretty_str('hx', self.hx),
            pretty_str('fx', self.fx),
            pretty_str('x_mean', self.x_mean),
            pretty_str('z_mean', self.z_mean),
            pretty_str('beta_n', self.beta_n)
            ])
