import numpy as np
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   sklearn.neighbors import KernelDensity
from   sklearn.model_selection import GridSearchCV
from   sklearn.covariance import EmpiricalCovariance
import warnings

import random
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
pi = np.pi


# General Constants
G = 6.6743 * 10**(-8)            # cm^3 / (g * s^2)
G_err = 1.5 * 10**(-12)

msun = 1.988409870698051*10**33           # g
msun_err = 4.468805426856864*10**28
mearth = 5.9722*10**27

rsun = 6.957 * 10**10           # cm
rsun_err = 1.4 * 10**7

rearth = 6.378137 * 10**8         # cm
rearth_err = 1.0 * 10**4

rhosun = 1.401
rhosun_err = 0.1401           # g / cc

day = 86400.                       # seconds


def calc_aor(P, rho):
    # P in sec; rho in g/cc
    
    aor = ( (P**2 * G * rho) / (3*np.pi) )**(1/3)
    return aor



### Winn (2010) ###
def calc_f_ew(e, w):
    # w in radians
    
    f_ew = (1 - e**2)**(1/2) / (1 + e*np.sin(w))
    return f_ew


def calc_T14_winn10(P, rho, b, ror, e, w):
    # P in sec; rho in g/cc; w in radians
    
    aor = calc_aor(P, rho)
    f_ew = calc_f_ew(e, w)
    
    T14 = P/np.pi * np.arcsin( ( (aor**2 - b**2) / ((1+ror)**2 - b**2) )**(-1/2) ) * f_ew 
    return T14 # in sec



### Kipping (2010a), (2014) ###
def calc_g_ew(e, w):
    # w in radians
    
    g_ew = (1 - e**2) / (1 + e*np.sin(w))
    return g_ew

def calc_T14_kipping10a(P, rho, b, ror, e, w):
    # P in sec; rho in g/cc; w in radians
    
    aor = calc_aor(P, rho)
    g_ew = calc_g_ew(e, w)
    
    T14 = P/np.pi * g_ew**2 / np.sqrt(1 - e**2) * np.arcsin( ( (aor**2 * g_ew**2 - b**2) / ((1+ror)**2 - b**2) )**(-1/2) )
    return T14 # in sec

def calc_rho_star(P, T14, b, ror, ecc, omega):
    # P in sec; T14 in sec; omega in rad
    
    oesw = 1 + ecc*np.sin(omega)
    oe2 = 1 - ecc**2
    oror = 1 + ror
    
    mult1 = ( oesw / oe2 )**3
    mult2 = 3*np.pi / (P**2 * G)
    mult3 = ( ( oror**2 - b**2 ) / np.sin( oesw**2 / oe2**(3/2) * T14*np.pi/P )**2 + b**2)**(3/2)
    
    product = mult1 * mult2 * mult3 # in g/cc
    
    if sum(np.isnan(product)) > 0:
        return np.nan_to_num(product, nan=0.0) 
    
    return product




### ecc-omega transit observability function ###
def get_e_omega_obs_priors(N, ecut):
    '''
    Get N random draws of ecc [0, ecut] and omega [-pi/2, 3pi/2],
    using the transit observability prior 
    (see: https://github.com/gjgilbert/notes/blob/main/calculate_e-omega_grid.ipynb)
    '''
    ngrid = 101
    ndraw = int(N)

    e_uni = np.linspace(0,ecut,ngrid)
    z_uni = np.linspace(0,1,ngrid)

    omega_grid = np.zeros((ngrid,ngrid))

    for i, e_ in enumerate(e_uni):
        x = np.linspace(-0.5*pi, 1.5*pi, int(1e4))
        y = (1 + e_*np.sin(x))/(2*pi)

        cdf = np.cumsum(y)
        cdf -= cdf.min()
        cdf = cdf/cdf.max()
        inv_cdf = interp1d(cdf, x)

        omega_grid[i] = inv_cdf(z_uni)

    RBS = RectBivariateSpline(e_uni, z_uni, omega_grid)

    e_draw = np.random.uniform(0, ecut, ndraw)
    z_draw = np.random.uniform(0, 1, ndraw)
    w_draw = RBS.ev(e_draw, z_draw)
    
    return e_draw, w_draw






### upgrading importance sampling function with RHOTILDE ####
def imp_sample_rhotilde(samples_mcmc, rho_star, norm=True, return_log=False, ecut=1.0, ew_obs_prior=False, distr='uniform', params=[]):
    '''
    Perform standard importance sampling from RHOTILDE --> {ECC, OMEGA}
    
    Args
    ----
    samples_mcmc [dataframe]: pandas dataframe of sampled data which includes: RHOTILDE
    rho_star [tuple]: values of the true stellar density and its uncertainty
    norm [bool]: True to normalize weights before output (default=True)
    return_log [bool]: True to return ln(weights) instead of weights (default=False)
    ecut [float]: number between 0 and 1 indicating the upper bound on the ecc prior (default 1.0)
    we_obs_prior [bool]: bool flag indicating whether or not to use the ecc-omega transit obs prior (default False)
    distr [str]: name of the distribution shape to sample ECC from; defaults to uniform
    params [list]: list of values to be used as parameters for the indicated distribution
    
    Output:
    weights [array]: importance sampling weights
    data [dataframe]: pandas dataframe containing all input and derived data, including: 
                      ECC: random values drawn from 0 to 'ecut' according to 'distr' and 'params'
                      OMEGA: random values drawn from -pi/2 to 3pi/2 (with transit obs prior if 'ew_obs_prior'=True)
                      RHOTILDE: inputs values
                      RHOSTAR: derived values
                      WEIGHTS (or LN_W): importance weights
    '''
    
    rho_tilde = samples_mcmc.RHOTILDE.values
    
    N = len(rho_tilde)
    
    if ew_obs_prior == True:
        ecc, omega = get_e_omega_obs_priors(N, ecut)
        
    else:
        if distr == 'uniform':
            ecc = np.random.uniform(0., ecut, N)

        elif distr == 'rayleigh':
            sigma = params[0]
            ecc = np.random.rayleigh(sigma, size=N)
            while np.any(ecc >= ecut):
                ecc[ecc >= ecut] = np.random.rayleigh(sigma, size=np.sum(ecc >= ecut))

        elif distr == 'beta':
            alpha_mu, beta_mu = params
            ecc = np.random.beta(alpha_mu, beta_mu, size=N)
            while np.any(ecc >= ecut):
                ecc[ecc >= ecut] = np.random.beta(alpha_mu, beta_mu, size=np.sum(ecc >= ecut))

        elif distr == 'half-gaussian':
            sigma = params[0]
            ecc = np.random.normal(loc=0, scale=sigma, size=N)
            while np.any((ecc >= ecut)|(ecc < 0)):
                ecc[(ecc >= ecut)|(ecc < 0)] = np.random.normal(loc=0, scale=sigma, size=np.sum((ecc >= ecut)|(ecc < 0)))

        omega = np.random.uniform(-0.5*np.pi, 1.5*np.pi, N)


    g = (1 + ecc * np.sin(omega)) / np.sqrt(1 - ecc ** 2)
    rho = rho_tilde / g ** 3

    log_weights = -0.5 * ((rho - rho_star[0]) / rho_star[1]) ** 2
    
    data = samples_mcmc.copy()
    data['ECC'] = ecc
    data['OMEGA'] = omega

    data['RHOSTAR'] = rho
    
    if return_log:
        data['LN_W'] = log_weights
        return log_weights, data
    
    else:
        weights = np.exp(log_weights - np.max(log_weights))

        if norm:
            weights /= np.sum(weights)
    
        data['WEIGHTS'] = weights
        return weights, data
    



### upgrading importance sampling function with DUR14 --> RHOSTAR ####
def imp_sample_rhostar(samples_mcmc, rho_star, norm=True, return_log=False, ecut=1.0, ew_obs_prior=False, distr='uniform', params=[]):
    '''
    Perform standard importance sampling from {IMPACT, ROR, PERIOD, DUR14} --> {ECC, OMEGA}
    
    Args
    ----
    samples_mcmc [dataframe]: pandas dataframe of sampled data which includes: IMPACT, ROR, PERIOD, DUR14
    rho_star [tuple]: values of the true stellar density and its uncertainty
    norm [bool]: True to normalize weights before output (default=True)
    return_log [bool]: True to return ln(weights) instead of weights (default=False)
    ecut [float]: number between 0 and 1 indicating the upper bound on the ecc prior (default 1.0)
    we_obs_prior [bool]: bool flag indicating whether or not to use the ecc-omega transit obs prior (default False)
    distr [str]: name of the distribution shape to sample ECC from; defaults to uniform
    params [list]: list of values to be used as parameters for the indicated distribution
    
    Output:
    weights [array]: importance sampling weights
    data [dataframe]: pandas dataframe containing all input and derived data, including: 
                      ECC: random values drawn from 0 to 'ecut' according to 'distr' and 'params'
                      OMEGA: random values drawn from -pi/2 to 3pi/2 (with transit obs prior if 'ew_obs_prior'=True)
                      IMPACT: inputs values
                      ROR: inputs values
                      PERIOD: inputs values
                      DUR14: inputs values
                      RHOSTAR: derived values
                      WEIGHTS (or LN_W): importance weights
    '''
    
    b = samples_mcmc.IMPACT.values
    ror = samples_mcmc.ROR.values
    P = samples_mcmc.PERIOD.values
    T14 = samples_mcmc.DUR14.values
    
    N = len(b)
    
    if ew_obs_prior == True:
        ecc, omega = get_e_omega_obs_priors(N, ecut)
        
    else:
        if distr == 'uniform':
            ecc = np.random.uniform(0., ecut, N)

        elif distr == 'rayleigh':
            sigma = params[0]
            ecc = np.random.rayleigh(sigma, size=N)
            while np.any(ecc >= ecut):
                ecc[ecc >= ecut] = np.random.rayleigh(sigma, size=np.sum(ecc >= ecut))

        elif distr == 'beta':
            alpha_mu, beta_mu = params
            ecc = np.random.beta(alpha_mu, beta_mu, size=N)
            while np.any(ecc >= ecut):
                ecc[ecc >= ecut] = np.random.beta(alpha_mu, beta_mu, size=np.sum(ecc >= ecut))

        elif distr == 'half-gaussian':
            sigma = params[0]
            ecc = np.random.normal(loc=0, scale=sigma, size=N)
            while np.any((ecc >= ecut)|(ecc < 0)):
                ecc[(ecc >= ecut)|(ecc < 0)] = np.random.normal(loc=0, scale=sigma, size=np.sum((ecc >= ecut)|(ecc < 0)))

        omega = np.random.uniform(-0.5*np.pi, 1.5*np.pi, N)
        
    
    if np.nanmedian(P) < 8000.:
        P *= 86400.
    
    if np.nanmedian(T14) < 100.:
        T14 *= 86400.
    
    rho_mcmc = calc_rho_star(P, T14, b, ror, ecc, omega)

    log_weights = -0.5 * ((rho_mcmc - rho_star[0]) / rho_star[1]) ** 2
    
    data = samples_mcmc.copy()
    data['ECC'] = ecc
    data['OMEGA'] = omega

    data['RHOSTAR'] = rho_mcmc
    
    if return_log:
        data['LN_W'] = log_weights
        return log_weights, data
    
    else:
        weights = np.exp(log_weights - np.max(log_weights))

        if norm:
            weights /= np.sum(weights)
        data['WEIGHTS'] = weights
        return weights, data






def get_bw(samples, num_max=1000):
    """
    Use cross-validation to determine KDE bandwidth for a set of 1D data samples
    Iteratively performs first coarse then fine estimation   
    
    Parameters
    ----------
    samples : array-like
        1D data samples
    num_max : int (default=1000)
        maximum number of samples to use during estimation
        
    Returns
    -------
    bw : float
        estimated bandwidth
    """
    N = len(samples)
    x = np.random.choice(samples, size=np.min([num_max,N]), replace=False)
    
    coarse_mesh = np.linspace((x.max()-x.min())/N, np.std(x), int(np.sqrt(N)))
    grid = GridSearchCV(KernelDensity(), {"bandwidth": coarse_mesh}, cv=5)
    grid.fit(x[:, None])
    
    fine_mesh = grid.best_params_["bandwidth"] + np.linspace(-1,1,int(np.sqrt(N)))*(coarse_mesh[1]-coarse_mesh[0])
    grid = GridSearchCV(KernelDensity(), {"bandwidth": fine_mesh}, cv=5)
    grid.fit(x[:, None])

    return grid.best_params_["bandwidth"]


def generate_synthetic_samples(samples, bandwidths, n_up, weights=None):
    """
    Use PDF Over-Sampling (PDFOS - Gao+ 2014) to generate synthetic samples
    
    Parameters
    ----------
    samples : ndarray, (N x M)
        array of data samples arranged N_samples, M_parameters
    bandwitdhs : array-like, (M)
        pre-estimated KDE bandwidths for each of M parameters
    n_up : int
        number of upsampled synthetic data points to generate
    weights : ndarray, (N x M)
        array of weights corresponding to samples
        
    Returns
    -------
    new_samples : ndarray
        array containing synthetic samples
    """
    n_samp, n_dim = samples.shape
    index = np.arange(0, n_samp, dtype='int')
    
    # we'll generate a few more samples than needed in anticipation of rejecting a few
    n_up101 = int(1.01*n_up)

    # naive resampling (used only to estimate covariance matrix)
    naive_resamples = samples[np.random.choice(index, p=weights, size=3*n_samp)]

    # compute empirical covariance matrix and lower Cholesky decomposition
    emp_cov = EmpiricalCovariance().fit(naive_resamples).covariance_
    L = np.linalg.cholesky(emp_cov)

    # scale each parameter by precomputed bandwidths so they have similar variance
    samples_scaled = (samples - np.mean(samples, axis=0)) / bandwidths

    # calculate synthetic samples following PDFOS (Gao+ 2014)
    random_index = np.random.choice(index, p=weights, size=n_up101)
    random_samples = samples_scaled[random_index]
    random_jitter = np.random.normal(0, 1, n_up101*n_dim).reshape(n_up101, n_dim)
    new_samples = random_samples + np.dot(L.T, random_jitter.T).T

    # rescale each parameter to invert previous scaling
    new_samples = new_samples*bandwidths + np.mean(samples, axis=0)
    
    # reject any synthetic samples pushed out of bounds of original samples
    bad = np.zeros(n_up101, dtype='bool')
    
    for i in range(n_dim):
        bad += (new_samples[:,i] < samples[:,i].min())
        bad += (new_samples[:,i] > samples[:,i].max())
        
    if np.sum(bad)/len(bad) > 0.01:
        warnings.warn("More than 1% of PDFOS generated samples were beyond min/max values of original samples")
    
    new_samples = new_samples[~bad]
    
    # only return n_up samples
    if new_samples.shape[0] >= n_up:
        return new_samples[:n_up]
    
    else:
        # use naive resampling to replace rejected samples
        replacement_samples = samples[np.random.choice(index, p=weights, size=n_up-new_samples.shape[0])]
    
        return np.vstack([new_samples, replacement_samples])



### original version of importance sampling ###
def imp_sample(rho_tilde, rho_star, norm=True, return_log=False, ecut=1.):
    '''
    Perform standard importance sampling from rho_tilde --> {e, omega}
    
    Args
    ----
    rho_tilde : [array] length-N array of sampled data for pseudo-density rho_tilde
    rho_star : [tuple] values of the true stellar density and its uncertainty
    norm : [bool] True to normalize weights before output (default=True)
    return_log :  [bool] True to return ln(weights) instead of weights (default=False)
    
    Output:
    weights [array]: importance sampling weights
    ecc [array]: random values drawn uniformly from 0 to 1, with array length = len(rho_array)
    omega [array]:random values drawn uniformly from -pi/2 to 3pi/2, with array length = len(rho_array)
    '''
    ecc = np.random.uniform(0., ecut, len(rho_tilde))
    omega = np.random.uniform(-0.5*np.pi, 1.5*np.pi, len(rho_tilde))

    g = (1 + ecc * np.sin(omega)) / np.sqrt(1 - ecc ** 2)
    rho = rho_tilde / g ** 3

    log_weights = -0.5 * ((rho - rho_star[0]) / rho_star[1]) ** 2
    
    if return_log:
        return log_weights, ecc, omega
    
    else:
        weights = np.exp(log_weights - np.max(log_weights))

        if norm:
            weights /= np.sum(weights)
    
        return weights, ecc, omega
    
    
