import sys
import os, fnmatch
import datetime

machine = 'cadence'
cadence = 30 # cadence of observations in minutes

if machine == 'hoffman2':
    sys.path.insert(0,'/usr/share/texmf/tex:/usr/share/texmf/tex/latex:/u/local/compilers/gcc/7.2.0/bin:/u/home/m/macdouga/miniconda3/bin:/u/home/m/macdouga/miniconda3/condabcondabin:/u/home/m/macdouga/miniconda3/pkgs/texlive-core-20180414-pl526h89d1741_1/bin:/u/home/m/macdouga/miniconda3/pkgs/texlive-core-20180414-pl526h89d1741_1')

import time
import lightkurve as lk
import exoplanet as xo
import pymc3 as pm
import pymc3_ext as pmx
import scipy
import theano
if machine == 'hoffman2':
    theano.config.allow_gc=True
    theano.config.scan.allow_fgc=True
    theano.config.scan.allow_output_prealloc=False

import theano.tensor as tt
from scipy import stats
import numpy as np
from numpy.random import normal, multivariate_normal
import uncertainties
from uncertainties import ufloat
import pandas as pd
from celerite2.theano import terms, GaussianProcess
from astropy.io import fits
import matplotlib.pyplot as plt
import arviz as az
from constants import *

import aesara_theano_fallback.tensor as T

from pymc3.distributions import generate_samples
from pymc3.distributions import transforms as tr






### General-use functions for astro calculations

def calc_aor(P, rho):
    # P in sec; rho in g/cc
    
    aor = ( (P**2 * G * rho) / (3*np.pi) )**(1/3)
    return aor


def get_sma(P, Ms):
    """
    Parameters
    ----------
    P : period [days]
    Ms : stellar mass [Solar masses]
    
    Returns
    -------
    sma : semimajor axis [Solar radii]
    """
    return Ms**(1/3)*(P/365.24)**(2/3)/RSAU


def calc_f_ew(e, w):
    # w in radians
    
    f_ew = (1 - e**2)**(1/2) / (1 + e*np.sin(w))
    return f_ew


def calc_T14(P, rho, b, ror, e, w):
    # P in sec; rho in g/cc; w in radians
    
    aor = calc_aor(P, rho)
    f_ew = calc_f_ew(e, w)
    
    T14 = P/np.pi * np.arcsin( ( (aor**2 - b**2) / ((1+ror)**2 - b**2) )**(-1/2) ) * f_ew 
    return T14 # in sec


def calc_noise(T14, N, ror, SNR, cadence=2):
    # T14 in sec; texp in min
    
    texp = (60.*cadence)
    
    noise = ( T14/texp * N )**(1/2) * ror**2 / SNR
    return noise


def simple_model(x, params):
    texp = np.min(np.diff(x))

    orbit = xo.orbits.KeplerianOrbit(rho_star=params['RHOSTAR'], period=params['PERIOD'], 
        t0=params['T0'], b=params['IMPACT'], ecc=params['ECC'], omega=params['OMEGA'])

    # Compute a limb-darkened light curve using starry
    light_curve = (
        xo.LimbDarkLightCurve(np.array([params['LD_U1'],params['LD_U2']]))
        .get_light_curve(orbit=orbit, r=params['ROR'], t=x, texp=texp)
        .eval()
    )

    return light_curve




### Functions to create sq(e)cos(w), sq(e)sin(w) from unit disk
# Modified to accept an upper bound that is not e = 1

upper_bound = np.sqrt(0.95)

class UnitDiskTransform(tr.Transform):
    """Transform the 2D real plane into a unit disk
    This will be especially useful for things like sampling in eccentricity
    vectors like ``e sin(w), e cos(w)``.
    """

    name = "unitdisk"

    def backward(self, y):
        return T.stack([y[0], y[1] * T.sqrt(upper_bound**2 - y[0] ** 2)])

    def forward(self, x):
        return T.stack([x[0], x[1] / T.sqrt(upper_bound**2 - x[0] ** 2)])

    def forward_val(self, x, point=None):
        return np.array([x[0], x[1] / np.sqrt(upper_bound**2 - x[0] ** 2)])

    def jacobian_det(self, y):
        return T.stack((T.zeros_like(y[0]), 0.5 * T.log(upper_bound**2 - y[0] ** 2)))


unit_disk = tr.Chain([UnitDiskTransform(), tr.Interval(-upper_bound, upper_bound)])

class UnitDisk(pm.Flat):
    """Two dimensional parameters constrianed to live within the unit disk
    This distribution is constrained such that the sum of squares along the
    zeroth axis will always be less than one. For example, in this code block:
    .. code-block:: python
        import aesara_theano_fallback.tensor as tt
        disk = UnitDisk("disk")
        radius = tt.sum(disk ** 2, axis=0)
    the tensor ``radius`` will always have a value in the range ``[0, 1)``.
    Note that the shape of this distribution must be two in the zeroth axis.
    """

    def __init__(self, *args, **kwargs):
        kwargs["transform"] = kwargs.pop("transform", unit_disk)

        # Make sure that the shape is compatible
        shape = kwargs["shape"] = kwargs.get("shape", 2)
        try:
            if list(shape)[0] != 2:
                raise ValueError("the first dimension should be exactly 2")
        except TypeError:
            if shape != 2:
                raise ValueError("the first dimension should be exactly 2")

        super().__init__(*args, **kwargs)

        # Work out some reasonable starting values for the parameters
        self._default = np.zeros(shape)

    def _random(self, size=None):
        sr = np.sqrt(np.random.uniform(0, upper_bound, size))
        theta = np.random.uniform(-np.pi, np.pi, size)
        return np.moveaxis(
            np.stack((sr * np.cos(theta), sr * np.sin(theta))), 0, 1
        )

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape[1:],
            broadcast_shape=self.shape[1:],
            size=size,
        )

    def logp(self, value):
        return tt.zeros_like(tt.sum(value, axis=0))






### ALL MODELS

# Set parameter prior boundaries

RORMIN = 0.005
RORMAX = 0.06

DURMIN = 0.005
DURMAX = 1.0

RHOMIN = 0.001
RHOMAX = 10000.

texp = (60.*cadence)/86400.



# Incorrect parameterization with rhostarcirc
def build_R(truths, data, texp=texp, rb_prior="uniform_logr", rhomin=RHOMIN, rhomax=RHOMAX, rmin=RORMIN, rmax=RORMAX):
    t = data["X"]
    y = data["Y"]
    yerr = truths["NOISE"]

    with pm.Model() as model_R:

        log_r = pm.Uniform("LN_ROR", lower=np.log(rmin), upper=np.log(rmax), testval=np.log(truths["ROR"]))
        r = pm.Deterministic("ROR", pm.math.exp(log_r))
        b = pm.Uniform("IMPACT", lower=0, upper=1+r, testval=truths["IMPACT"])
        
        # adjustment term makes samples uniform in the (r,b) plane
        adj = pm.Potential("ADJ", T.log(1+r) + T.log(r))
        
        # enforce desired prior on (r,b)
        if rb_prior == "espinoza18":
            r_marginal = 0.0
        elif rb_prior == "uniform_r":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r))
        elif rb_prior == "uniform_logr":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r) - T.log(r))
        else:
            raise ValueError("rb_prior must be 'espinoza18', 'uniform_r', or 'uniform_logr'")
            
        
        # draw other parameters
        ###mean =  pm.Normal("mean", mu=0.0, sd=1.0)
        ###log_jit = pm.Normal("log_jit", mu=np.log(yerr/1e6), sd=10.0)
        u = xo.QuadLimbDark("LD_U", testval=[truths["LD_U1"], truths["LD_U2"]])
        t0 = pm.Normal("T0", mu=truths["T0"], sd=0.1)

        ###
        rho = pm.Uniform("RHOTILDE", lower=rhomin, upper=rhomax, testval=truths["RHOSTAR"])
        ###
        
        # orbit and lightcurve
        starrystar = xo.LimbDarkLightCurve(u)
        orbit = xo.orbits.KeplerianOrbit(period=truths["PERIOD"], t0=t0, b=b, ror=r, rho_star=rho)
        light_curve = starrystar.get_light_curve(orbit=orbit, t=t, r=r, texp=texp)[:,0]
        
        # track orbital period and stellar density
        per = pm.Deterministic("PERIOD", orbit.period)

        # Likelihood
        pm.Normal("OBS", mu=light_curve, sigma=yerr, observed=y) ###pm.math.sqrt(yerr**2 + pm.math.exp(log_jit)), observed=y)

    return model_R






# Standard fitting with duration
def build_S(truths, data, texp=texp, rb_prior="uniform_logr", durmin=DURMIN, durmax=DURMAX, rmin=RORMIN, rmax=RORMAX):
    t = data["X"]
    y = data["Y"]
    yerr = truths["NOISE"]

    with pm.Model() as model_S:

        log_r = pm.Uniform("LN_ROR", lower=np.log(rmin), upper=np.log(rmax), testval=np.log(truths["ROR"]))
        r = pm.Deterministic("ROR", pm.math.exp(log_r))
        b = pm.Uniform("IMPACT", lower=0, upper=1+r, testval=truths["IMPACT"])
        
        # adjustment term makes samples uniform in the (r,b) plane
        adj = pm.Potential("ADJ", T.log(1+r) + T.log(r))
        
        # enforce desired prior on (r,b)
        if rb_prior == "espinoza18":
            r_marginal = 0.0
        elif rb_prior == "uniform_r":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r))
        elif rb_prior == "uniform_logr":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r) - T.log(r))
        else:
            raise ValueError("rb_prior must be 'espinoza18', 'uniform_r', or 'uniform_logr'")
            
        
        # draw other parameters
        ###mean =  pm.Normal("mean", mu=0.0, sd=1.0)
        ###log_jit = pm.Normal("log_jit", mu=np.log(yerr/1e6), sd=10.0)
        u = xo.QuadLimbDark("LD_U", testval=[truths["LD_U1"], truths["LD_U2"]])
        t0 = pm.Normal("T0", mu=truths["T0"], sd=0.1)
        log_dur = pm.Uniform("LN_DUR14", lower=np.log(durmin), upper=np.log(durmax), testval=np.log(truths["DUR14"]))
        
        # orbit and lightcurve
        starrystar = xo.LimbDarkLightCurve(u)
        orbit = xo.orbits.KeplerianOrbit(period=truths["PERIOD"], t0=t0, b=b, ror=r, duration=pm.math.exp(log_dur))
        light_curve = starrystar.get_light_curve(orbit=orbit, t=t, r=r, texp=texp)[:,0]
        
        # track orbital period and stellar density
        per = pm.Deterministic("PERIOD", orbit.period)
        rho = pm.Deterministic("RHOTILDE", orbit.rho_star)

        # Likelihood
        pm.Normal("OBS", mu=light_curve, sigma=yerr, observed=y) ###pm.math.sqrt(yerr**2 + pm.math.exp(log_jit)), observed=y)

    return model_S




### Umbrella Sampling ###

# Non-grazing
def build_N(truths, data, texp=texp, rb_prior="uniform_logr", durmin=DURMIN, durmax=DURMAX, rmin=RORMIN, rmax=RORMAX):
    t = data["X"]
    y = data["Y"]
    yerr = truths["NOISE"]

    with pm.Model() as model_N:

        # draw (r,b)
        log_r = pm.Uniform("LN_ROR", lower=np.log(rmin), upper=np.log(rmax), testval=np.log(truths["ROR"]))
        r = pm.Deterministic("ROR", T.exp(log_r))
        b = pm.Uniform("IMPACT", lower=0, upper=1-r, testval=truths["IMPACT"])
        g = (1-b)/r
        
        # this adjustment term makes samples uniform in the (r,b) plane
        adj = pm.Potential("ADJ", T.log(1-r) + T.log(r))
        
        # enforce desired prior on (r,b)
        if rb_prior == "espinoza18":
            r_marginal = 0.0
        elif rb_prior == "uniform_r":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r))
        elif rb_prior == "uniform_logr":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r) - T.log(r))
        else:
            raise ValueError("rb_prior must be 'espinoza18', 'uniform_r', or 'uniform_logr'")
        

        # draw other parameters
        ###mean =  pm.Normal("mean", mu=0.0, sd=1.0)
        ###log_jit = pm.Normal("log_jit", mu=np.log(yerr/1e6), sd=10.0)
        u = xo.QuadLimbDark("LD_U", testval=[truths["LD_U1"], truths["LD_U2"]])
        t0 = pm.Normal("T0", mu=truths["T0"], sigma=0.1)
        log_dur = pm.Uniform("LN_DUR14", lower=np.log(durmin), upper=np.log(durmax), testval=np.log(truths["DUR14"]))    
        
        # umbrella bias
        norm = 1/rmin - 1.5
        psi = pm.Potential("PSI_POT", T.log(T.switch(T.lt(g,2), g-1, 1.0)/norm))
        pm.Deterministic("PSI", psi)
        
        # orbit and lightcurve
        starrystar = xo.LimbDarkLightCurve(u)
        orbit = xo.orbits.KeplerianOrbit(period=truths["PERIOD"], t0=t0, b=b, ror=r, duration=pm.math.exp(log_dur))
        light_curve = starrystar.get_light_curve(orbit=orbit, t=t, r=r, texp=texp)[:,0]
        
        # track stellar density
        rho = pm.Deterministic("RHOTILDE", orbit.rho_star)
        per = pm.Deterministic("PERIOD", orbit.period)
        
        # Likelihood
        pm.Normal("OBS", mu=light_curve, sigma=yerr, observed=y) ###pm.math.sqrt(yerr**2 + pm.math.exp(log_jit)), observed=y)

    return model_N



# Transition
def build_T(truths, data, texp=texp, rb_prior="uniform_logr", durmin=DURMIN, durmax=DURMAX, rmin=RORMIN, rmax=RORMAX):
    t = data["X"]
    y = data["Y"]
    yerr = truths["NOISE"]
    
    with pm.Model() as model_T:

        # draw (r,gamma)
        log_r = pm.Uniform("LN_ROR", lower=np.log(rmin), upper=np.log(rmax), testval=np.log(truths["ROR"]))
        r = pm.Deterministic("ROR", T.exp(log_r))
        g = pm.Uniform("GAMMA", lower=0, upper=T.switch(r < 0.5, 2, 1/r), testval=1.0)
        b = pm.Deterministic("IMPACT", 1-g*r)
        
        # Jacobian for (r,b) --> (r,gamma)
        jac = pm.Potential("JAC", T.log(1/r))
        
        # this adjustment term makes samples uniform in the (r,b) plane
        adj = pm.Potential("ADJ", 2*T.log(r) + T.switch(r < 0.5, T.log(2*r), 0.0))
        
        # enforce desired prior on (r,b)
        if rb_prior == "espinoza18":
            r_marginal = 0.0
        elif rb_prior == "uniform_r":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r))
        elif rb_prior == "uniform_logr":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r) - T.log(r))
        else:
            raise ValueError("rb_prior must be 'espinoza18', 'uniform_r', or 'uniform_logr'")
        
        
        # draw other parameters
        ###mean =  pm.Normal("mean", mu=0.0, sd=1.0)
        ###log_jit = pm.Normal("log_jit", mu=np.log(yerr/1e6), sd=10.0)   
        u = xo.QuadLimbDark("LD_U", testval=[truths["LD_U1"], truths["LD_U2"]])
        t0 = pm.Normal("T0", mu=truths["T0"], sigma=0.1)
        log_dur = pm.Uniform("LN_DUR14", lower=np.log(durmin), upper=np.log(durmax), testval=np.log(truths["DUR14"]))
        
        # umbrella bias
        norm = 1.0
        psi = pm.Potential("PSI_POT", T.log(T.switch(T.lt(g,1), g, 2-g))/norm)
        pm.Deterministic("PSI", psi)


        # orbit and lightcurve
        starrystar = xo.LimbDarkLightCurve(u)
        orbit = xo.orbits.KeplerianOrbit(period=truths["PERIOD"], t0=t0, b=b, ror=r, duration=pm.math.exp(log_dur))
        light_curve = starrystar.get_light_curve(orbit=orbit, t=t, r=r, texp=texp)[:,0]
        
        # track stellar density
        rho = pm.Deterministic("RHOTILDE", orbit.rho_star)
        per = pm.Deterministic("PERIOD", orbit.period)
        
        # Likelihood
        pm.Normal("OBS", mu=light_curve, sigma=yerr, observed=y) ###pm.math.sqrt(yerr**2 + pm.math.exp(log_jit)), observed=y)

    return model_T





# Grazing (r, b)
def build_Grb(truths, data, texp=texp, rb_prior="uniform_logr", durmin=DURMIN, durmax=DURMAX, rmin=RORMIN, rmax=RORMAX):
    t = data["X"]
    y = data["Y"]
    yerr = truths["NOISE"]
    
    with pm.Model() as model_Grb:

        # draw (r,b)
        log_r = pm.Uniform("LN_ROR", lower=np.log(rmin), upper=np.log(rmax), testval=np.log(truths["ROR"]))
        r = pm.Deterministic("ROR", T.exp(log_r))
        b = pm.Uniform("IMPACT", lower=1-r, upper=1+r, testval=1.0)
        
        g = pm.Deterministic("GAMMA", (1-b)/r)
        ###lam = pm.Deterministic("lam", r**2 + (1-b)*r)
        ###log_lam = pm.Deterministic("log_lam", T.log(lam))

        # this adjustment term makes samples uniform in the (r,b) plane
        adj = pm.Potential("ADJ", T.log(2*r) + T.log(r))


        # enforce desired prior on (r,b)
        if rb_prior == "espinoza18":
            r_marginal = 0.0
        elif rb_prior == "uniform_r":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r))
        elif rb_prior == "uniform_logr":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r) - T.log(r))
        else:
            raise ValueError("rb_prior must be 'espinoza18', 'uniform_r', or 'uniform_logr'")

            
        # draw other parameters
        ###mean =  pm.Normal("mean", mu=0.0, sd=1.0)
        ###log_jit = pm.Normal("log_jit", mu=np.log(yerr/1e6), sd=10.0)   
        u = xo.QuadLimbDark("LD_U", testval=[truths["LD_U1"], truths["LD_U2"]])
        t0 = pm.Normal("T0", mu=truths["T0"], sigma=0.1)
        log_dur = pm.Uniform("LN_DUR14", lower=np.log(durmin), upper=np.log(durmax), testval=np.log(truths["DUR14"]))
        
        # umbrella bias
        norm = 1.0
        psi = pm.Potential("PSI_POT", T.log(T.switch(T.lt(g,0), 1+g, 1-g)/norm))
        pm.Deterministic("PSI", psi)
        
        # orbit and lightcurve
        starrystar = xo.LimbDarkLightCurve(u)
        orbit = xo.orbits.KeplerianOrbit(period=truths["PERIOD"], t0=t0, b=b, ror=r, duration=pm.math.exp(log_dur))
        light_curve = starrystar.get_light_curve(orbit=orbit, t=t, r=r, texp=texp)[:,0]
        
        # track stellar density
        rho = pm.Deterministic("RHOTILDE", orbit.rho_star)
        per = pm.Deterministic("PERIOD", orbit.period)
        
        # Likelihood
        pm.Normal("OBS", mu=light_curve, sigma=yerr, observed=y) ###pm.math.sqrt(yerr**2 + pm.math.exp(log_jit)), observed=y)

    return model_Grb




# Grazing (l, g)
def build_G(truths, data, texp=texp, rb_prior="uniform_logr", durmin=DURMIN, durmax=DURMAX, rmin=RORMIN, rmax=RORMAX):
    t = data["X"]
    y = data["Y"]
    yerr = truths["NOISE"]
    
    with pm.Model() as model_G:
        
        # draw (lambda, gamma)
        g = pm.Uniform("GAMMA", lower=-0.99, upper=1.0, testval=0.0)
        log_lam = pm.Uniform("LN_LAM", lower=np.log((g+1)*rmin**2), upper=np.log((g+1)*rmax**2), testval=np.log(0.001))
        lam = pm.Deterministic("LAMBDA", T.exp(log_lam))
        
        r = pm.Deterministic("ROR", pm.math.sqrt(lam/(g+1)))
        log_r = pm.Deterministic("LN_ROR", T.log(r))
        b = pm.Deterministic("IMPACT", 1-g*r)

        # Jacobian for (r,b) --> (lambda,gamma)
        jac = pm.Potential("JAC", T.log(2 + 2*g))
        
        # this adjustment term makes samples uniform in the (r,b) plane
        adj = pm.Potential("ADJ", -T.log(2 + 2*g) + 2*T.log(r))


        # enforce desired prior on (r,b)
        if rb_prior == "espinoza18":
            r_marginal = 0.0
        elif rb_prior == "uniform_r":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r))
        elif rb_prior == "uniform_logr":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r) - T.log(r))
        else:
            raise ValueError("rb_prior must be 'espinoza18', 'uniform_r', or 'uniform_logr'")

            
        # draw other parameters
        ###mean =  pm.Normal("mean", mu=0.0, sd=1.0)
        ###log_jit = pm.Normal("log_jit", mu=np.log(yerr/1e6), sd=10.0)   
        u = xo.QuadLimbDark("LD_U", testval=[truths["LD_U1"], truths["LD_U2"]])
        t0 = pm.Normal("T0", mu=truths["T0"], sigma=0.1)
        log_dur = pm.Uniform("LN_DUR14", lower=np.log(durmin), upper=np.log(durmax), testval=np.log(truths["DUR14"]))
        
        # umbrella bias
        norm = 1.0
        psi = pm.Potential("PSI_POT", T.log(T.switch(T.lt(g,0), 1+g, 1-g)/norm))
        pm.Deterministic("PSI", psi)
        
        # orbit and lightcurve
        starrystar = xo.LimbDarkLightCurve(u)
        orbit = xo.orbits.KeplerianOrbit(period=truths["PERIOD"], t0=t0, b=b, ror=r, duration=pm.math.exp(log_dur))
        light_curve = starrystar.get_light_curve(orbit=orbit, t=t, r=r, texp=texp)[:,0]
        
        # track stellar density
        rho = pm.Deterministic("RHOTILDE", orbit.rho_star)
        per = pm.Deterministic("PERIOD", orbit.period)
        
        # Likelihood
        pm.Normal("OBS", mu=light_curve, sigma=yerr, observed=y) ###pm.math.sqrt(yerr**2 + pm.math.exp(log_jit)), observed=y)

    return model_G



# Directly fitting in e and omega (EOR model)
def build_EOR(truths, data, texp=texp, rb_prior="uniform_logr", durmin=DURMIN, durmax=DURMAX, rmin=RORMIN, rmax=RORMAX):
    t = data["X"]
    y = data["Y"]
    yerr = truths["NOISE"]
    
    with pm.Model() as model_EOR:

        log_r = pm.Uniform("LN_ROR", lower=np.log(rmin), upper=np.log(rmax), testval=np.log(truths["ROR"]))
        r = pm.Deterministic("ROR", pm.math.exp(log_r))
        b = pm.Uniform("IMPACT", lower=0, upper=1+r, testval=truths["IMPACT"])
        
        # adjustment term makes samples uniform in the (r,b) plane
        adj = pm.Potential("ADJ", T.log(1+r) + T.log(r))
        
        # enforce desired prior on (r,b)
        if rb_prior == "espinoza18":
            r_marginal = 0.0
        elif rb_prior == "uniform_r":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r))
        elif rb_prior == "uniform_logr":
            r_marginal = pm.Potential("r_marginal", -T.log(1+r) - T.log(r))
        else:
            raise ValueError("rb_prior must be 'espinoza18', 'uniform_r', or 'uniform_logr'")
            
        
        # draw other parameters
        ###mean =  pm.Normal("mean", mu=0.0, sd=1.0)
        ###log_jit = pm.Normal("log_jit", mu=np.log(yerr/1e6), sd=10.0)
        u = xo.QuadLimbDark("LD_U", testval=[truths["LD_U1"], truths["LD_U2"]])
        t0 = pm.Normal("T0", mu=truths["T0"], sd=0.1)
        
        ###
        rho = pm.Normal("RHOSTAR", mu=truths["RHOSTAR"], sd=truths["RHOSTARE"])
        ###log_rho = pm.Normal("log_rho", mu=np.log(truths["rho"]), sd=np.log(truths["rho"]*0.1))

        ecs = UnitDisk("ECS", testval=np.array([0.1, 0.1]))
        ecc = pm.Deterministic("ECC", T.sum(ecs ** 2))
        omega = pm.Deterministic("OMEGA", T.arctan2(ecs[1], ecs[0]))
        ###
        
        # orbit and lightcurve
        starrystar = xo.LimbDarkLightCurve(u)
        orbit = xo.orbits.KeplerianOrbit(period=truths["PERIOD"], t0=t0, b=b, ror=r, rho_star=rho, ecc=ecc, omega=omega)
        light_curve = starrystar.get_light_curve(orbit=orbit, t=t, r=r, texp=texp)[:,0]
        
        # track orbital period and stellar density
        per = pm.Deterministic("PERIOD", orbit.period)
        
        # Likelihood
        pm.Normal("OBS", mu=light_curve, sigma=yerr, observed=y) ###pm.math.sqrt(yerr**2 + pm.math.exp(log_jit)), observed=y)

    return model_EOR




def build_models(truths, data, texp=texp, rb_prior="uniform_logr", durmin=DURMIN, durmax=DURMAX, rmin=RORMIN, rmax=RORMAX, rhomin=RHOMIN, rhomax=RHOMAX):

    model_R = build_R(truths, data, texp=texp, rb_prior="uniform_logr", rhomin=rhomin, rhomax=rhomax, rmin=rmin, rmax=rmax)
    model_S = build_S(truths, data, texp=texp, rb_prior="uniform_logr", durmin=durmin, durmax=durmax, rmin=rmin, rmax=rmax)
    model_N = build_N(truths, data, texp=texp, rb_prior="uniform_logr", durmin=durmin, durmax=durmax, rmin=rmin, rmax=rmax)
    model_T = build_T(truths, data, texp=texp, rb_prior="uniform_logr", durmin=durmin, durmax=durmax, rmin=rmin, rmax=rmax)
    model_G = build_G(truths, data, texp=texp, rb_prior="uniform_logr", durmin=durmin, durmax=durmax, rmin=rmin, rmax=rmax)
    ###model_Gly = build_Gly(truths, data, texp=texp, rb_prior="uniform_logr", durmin=durmin, durmax=durmax, rmin=rmin, rmax=rmax)
    model_EOR = build_EOR(truths, data, texp=texp, rb_prior="uniform_logr", durmin=durmin, durmax=durmax, rmin=rmin, rmax=rmax)

    return [model_R, model_S, model_N, model_T, model_G, model_EOR] # model_Gly, 






def fit_models(params, models, DIR, ID, model_tags=['N', 'T', 'G', 'EOR'], tune=[15000,5000], draws=[5000,2000], chains=[2,2], cores=1, target_accept=0.95, return_inferencedata=True, discard_tuned_samples=True):

    all_tags = ['R', 'S', 'N', 'T', 'G', 'EOR'] # 'Gly', 

    model_dict = {}
    for i in range(len(all_tags)):
        model_dict[all_tags[i]] = models[i]

    inputs_init = [tune, draws, chains]
    inputs = []
    for i in inputs_init:
        if type(i) == int:
            inputs.append([i])
        else:
            inputs.append(i)

    ew_inputs = []
    for i in inputs:
        if len(i) != 2:
            ew_inputs.append(i[0])
        else:
            ew_inputs.append(i[1])

    summaries = {}
    traces = {}

    for model_tag in model_tags:
        model = model_dict[model_tag]
        if model_tag == 'EOR':
            tune=ew_inputs[0]
            draws=ew_inputs[1]
            chains=ew_inputs[2]
        else:
            tune=inputs[0][0]
            draws=inputs[1][0]
            chains=inputs[2][0]

        summ, trace =  apply_model(model, DIR, ID, model_tag, tune=tune, draws=draws, chains=chains, 
                                cores=cores, target_accept=target_accept, return_inferencedata=return_inferencedata, 
                                discard_tuned_samples=discard_tuned_samples)
        summaries[model_tag] = summ
        traces[model_tag] = trace

    return summaries, traces



def apply_model(model, DIR, ID, model_tag, tune=20000, draws=5000, chains=2, cores=1, target_accept=0.95, return_inferencedata=True, discard_tuned_samples=True):
    with model:
        trace = pmx.sample(tune=tune, draws=draws, chains=chains, cores=cores,
                           target_accept=target_accept, return_inferencedata=return_inferencedata, 
                           discard_tuned_samples=discard_tuned_samples)

    summary = pm.summary(trace)
    summary.columns = [col.upper().replace('%','').replace('_3','_03') for col in list(summary.columns)]

    label_dict = {
        'LD_U[0]': 'LD_U1',
        'LD_U[1]': 'LD_U2',
        'ECS[0]': 'SQE_COSO',
        'ECS[1]': 'SQE_SINO'
    }

    labels = []
    for idx in list(summary.index):
        if idx in label_dict.keys():
            labels.append(label_dict[idx])
        else:
            labels.append(idx)

    summary.index = labels

    # Traceplot
    if 'ECC' in list(summary.index):
        var_names = ['LN_ROR', 'IMPACT', 'ECC', 'OMEGA', 'RHOSTAR', 'LD_U']
    else:
        var_names = ['LN_ROR', 'IMPACT', 'RHOTILDE', 'LD_U']
    fig1 = pm.traceplot(trace, var_names=var_names, compact=False)
    plt.rc('text', usetex=False)
    figall = fig1[0][0].figure
    figall.tight_layout()
    figall.savefig(DIR + ID + "_" + model_tag + "-traceplot.png")
    plt.close()

    return summary, trace



