import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, leastsq

def linear_fit(args, x, y, num):
    m, b = args
    fit = m*x+b#b * x**m 
    return np.nansum((y-fit)**2/num**2)

def linear(args, x):
    m, b = args
    fit = m*x+b#b * x**m
    return fit

def power_law(args, x):
    m, b = args
    fit = b * x**m
    return fit

def linear_resid(args, x, y, num):
    m, b = args
    fit = m*x+b#b * x**m 
    return (y-fit)/num

def power_law_resid(args, x, y, num):
    m, b = args
    fit = b * x**m 
    return (y-fit)/num

## FUNCTIONS FOR MCMC FITS ##

def log_probability(theta, x, y, yerr1, yerr2):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr1, yerr2)

def log_prior(theta):
    #m, b, log_f = theta
    m, b = theta
    if -5.0 < m < 0 and -5 < b < 10:# and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf

def log_likelihood(theta, x, y, yerr1, yerr2):
    #m, b, log_f = theta
    m, b = theta
    model = m * x + b
    o = (y-model)
    yerr = np.sqrt(yerr1**2 + yerr2**2)
    return  -0.5 * np.sum(o**2 / yerr**2) 

def run_mcmc(x, y, lowlim, upplim, initguess, nwalkers=300, nsteps=5000, mask=None,
             plot_chains=True, plot_corner=True):

    q = y > 0

    pos = np.array([initguess[0], np.log10(initguess[1])]) + 1e-4 * np.random.randn(nwalkers,2)
    nwalkers, ndim = pos.shape

    if mask is None:
        mask = np.arange(0,len(x),1,dtype=int)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(np.log10(x[1:][q][mask]),
                                          np.log10(y[q][mask]),
                                          np.log10(lowlim[q][mask])-np.log10(y[q][mask]),
                                          np.log10(y[q][mask])-np.log10(upplim[q][mask])
                                         )
                                   )
    sampler.run_mcmc(pos, nsteps,progress=True)
    
    if plot_chains:

        fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = ["m", "b"]
        for n in range(ndim):
            ax = axes[n]
            ax.plot(samples[:, :, n], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[n])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            
        axes[-1].set_xlabel("step number")

        plt.show()
    
    flat_samples = sampler.get_chain(discard=800, thin=15, flat=True)
    

    if plot_corner:
        fig = corner.corner( flat_samples, labels=['m', 'b'],
                             truths=[initguess[0], initguess[1]])
        plt.show()

    for j in range(ndim-1):
        mcmc = np.percentile(flat_samples[:, j], [16, 50, 84])
        q = np.diff(mcmc)

    return flat_samples, [mcmc[1], q[0], q[1]]
