import numpy as np
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   sklearn.neighbors import KernelDensity
from   sklearn.model_selection import GridSearchCV
from   sklearn.covariance import EmpiricalCovariance
import warnings


def weighted_mean(x, w):
    return np.average(x, weights=w)


def weighted_std(x, w):
    mean = weighted_mean(x, w)
    return np.sqrt(np.average((x-mean)**2, weights=w))


def weighted_var(x, w):
    mean = weighted_mean(x, w)
    return np.average((x-mean)**2, weights=w)


def weighted_percentile(a, q, w=None):
    """
    Compute the weighted q-th percentile of the data. 
    Similar to np.percentile, but allows for weights. Axis slicing not supported.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100 inclusive.
    """
    a = np.array(a)
    q = np.array(q)
    
    if w is None:
        return np.percentile(a,q)
    
    else:
        w = np.array(w)
        w /= np.sum(w)
        
        assert np.all(q >= 0) and np.all(q <= 100), "quantiles should be in [0, 100]"

        order = np.argsort(a)
        a = a[order]
        w = w[order]

        weighted_quantiles = np.cumsum(w) - 0.5 * w
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    
        return np.interp(q/100., weighted_quantiles, a)


def anderson_2samp(x1, x2):
    """
    Adapts scipy.stats.anderson_ksamp to avoid floor/ceiling on p-values
    Makes input syntax similar to scipy.stats.ks_2samp
    """
    A, cv, p = stats.anderson_ksamp([x1, x2])

    alpha = np.array([0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001])
    
    res_fxn = lambda theta, x, y: y - 1/(1 + theta[0]*np.exp(x/theta[1]))
    
    fit, success = op.leastsq(res_fxn, [0.25, 1], args=(cv, alpha))
    
    z = np.linspace(-2,6)
    p_new = 1/(1 + fit[0]*np.exp(A/fit[1]))
    
    return A, p_new


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



def imp_sample(rho_tilde, rho_star, norm=True, return_log=False):
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
    ecc = np.random.uniform(0., 1., len(rho_tilde))
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
    
    

def boxcar_smooth(x, winsize, passes=2):
    """
    Smooth a data array with a sliding boxcar filter
    
    Parameters
    ----------
        x : ndarray
            data to be smoothed
        winsize : int
            size of boxcar window
        passes : int
            number of passes (default=2)
            
    Returns
    -------
        xsmooth : ndarray
            smoothed data array,same size as input 'x'
    """
    win = sig.boxcar(winsize)/winsize
    xsmooth = np.pad(x, (winsize, winsize), mode='reflect')

    for i in range(passes):
        xsmooth = sig.convolve(xsmooth, win, mode='same')
    
    xsmooth = xsmooth[winsize:-winsize]
    
    return xsmooth