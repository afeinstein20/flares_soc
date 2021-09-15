import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize, leastsq


################################
## FUNCTIONS FOR GENERIC FITS ##
################################ 

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


##############################
## FUNCTIONS FOR INIT GUESS ##
##############################

def amp_slope_fit(data, bins, i=0, j=-1, plot=True, get_err=True):
    
    n, _ = np.histogram(data['amp']*100, bins=bins)
    y, binedges, _ = plt.hist(data['amp']*100, bins=bins,
                          weights=np.full(len(data['amp']),
                                          1.0/np.nansum(data['weights'])),
                          alpha=0.4)
    plt.close()
    x = binedges[1:] + 0.0
    logx = np.log10(x)
    logn = np.log10(n)
    q = logn > 0

    results = minimize(linear_fit, x0=[-2.5, 7],
                       args=(logx[q][i:j-1]-np.diff(logx[q][i:j])/2., 
                             logn[q][i:j-1], np.sqrt(logn[q][i:j-1]) ), 
                       bounds=( (-10.0, 10.0), (-100, 100)),
                       method='L-BFGS-B', tol=1e-8)
    
    results.x[1] = 10**results.x[1]

    results2 = leastsq(power_law_resid, results.x,
                       args=(x[q][i:j-1]-np.diff(x[q][i:j])/2., 
                             n[q][i:j-1], 
                             np.sqrt(n[q][i:j-1]) ),
                       full_output=True)
    
    fit_params = results2[0]

    if get_err:
        slope_err = np.sqrt(results2[1][0][0])
        
    model = linear([fit_params[0], np.log10(fit_params[1])], logx)

    if plot:
        plt.plot(logx[i:j], np.log10(n[i:j]), '.', c='k')
        plt.plot(logx[i:j], linear([-2.5, 7], logx[i:j]), '--', c='w', linewidth=3)
        plt.plot(logx, model, c='r')
        plt.title('{} $\pm$ {}'.format(np.round(fit_params[0],2),
                                   np.round(slope_err,2)))
        plt.show()

    if get_err:
        return fit_params[0], slope_err, n, results.x[1]
    else:
        return fit_params[0], n, results.x[1]


#############################
## FUNCTIONS FOR MCMC FITS ##
#############################

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

    q = y[mask] > 0

    pos = np.array([initguess[0], np.log10(initguess[1])]) + 1e-4 * np.random.randn(nwalkers,2)
    nwalkers, ndim = pos.shape

    if mask is None:
        mask = np.arange(0,len(y)-1,1,dtype=int)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=(np.log10(x[1:][mask][q]),
                                          np.log10(y[mask][q]),
                                          np.log10(lowlim[mask][q])-np.log10(y[mask][q]),
                                          np.log10(y[mask][q])-np.log10(upplim[mask][q])
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


    mcmc_slope = np.percentile(flat_samples[:, 0], [16, 50, 84])
    q = np.diff(mcmc_slope)

    mcmc_offset = np.percentile(flat_samples[:, 1], [16, 50, 84])
    qo = np.diff(mcmc_offset)
        

    return flat_samples, [mcmc_slope[1], q[0], q[1]], [mcmc_offset[1], qo[0], qo[1]]
