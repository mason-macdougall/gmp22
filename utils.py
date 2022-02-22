import numpy as np
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   sklearn.neighbors import KernelDensity
from   sklearn.model_selection import GridSearchCV
from   sklearn.covariance import EmpiricalCovariance


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


def generate_synthetic_samples(samples, bandwidths, n_up):
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
        
    Returns
    -------
    new_samples : ndarray
        array containing synthetic samples
    """
    n_samp, n_dim = samples.shape
    
    # compute empirical covariance matrix and lower Cholesky decomposition
    emp_cov = EmpiricalCovariance().fit(samples).covariance_
    L = np.linalg.cholesky(emp_cov)

    # scale each parameter by precompputed bandwidths so they have similar variance
    samples_scaled = (samples - np.mean(samples, axis=0)) / bandwidths

    # calculate synthetic samples following PDFOS (Gao+ 2014)
    random_index = np.random.randint(0, n_samp, n_up)
    random_samples = samples_scaled[random_index]
    random_weights = np.random.normal(0, 1, n_up*n_dim).reshape(n_up, n_dim)
    new_samples = random_samples + np.dot(L.T, random_weights.T).T

    # rescale each parameter to invert previous scaling
    return new_samples*bandwidths + np.mean(samples, axis=0)


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