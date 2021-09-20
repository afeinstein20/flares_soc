import numpy as np
import matplotlib.pyplot as plt
from selection import read_mamajek
from matplotlib.colors import LinearSegmentedColormap

def hex_to_rgb(h):
    if '#' in h:
        h = h.lstrip('#')   
    hlen = int(len(h))
    rgb = tuple(int(h[i:int(i+hlen/3)], 16) / 255.0 for i in range(0, hlen, int(hlen/3)))
    return rgb

def make_cmap(clist, name='sequential'):
    rgb_tuples = []

    for c in clist:
        rgb_tuples.append(hex_to_rgb(c))

    cm = LinearSegmentedColormap.from_list(
            name, rgb_tuples, N=2048)
    return cm

def flarerate_cmap():
    clist0 = np.array(['EA8F3C', 'EB6A41', '69469D', '241817'])
    cm = make_cmap(clist0, name='halloween')
    return cm

def ruwe_cmap():
    clist1 = np.array(['66C6C6', '2B8D9D', '19536C', '123958', '121422'])
    cm = make_cmap(clist1, name='ocean')
    return cm

def present_cmap():
    clist1 = np.array(['FFFFFF', '8cb0ca','487b96','182436'])
    cm = make_cmap(clist1, name='misc')
    return cm

def discrete_cmap():
    clist1 = np.array(['#60374c', '#a96388', '#c49ab6', 
                       '#8cb0ca', '#487b96', '#3e6474'])
    return 0, clist1, 0


def parula_cmap(histplots):
    parula = np.load('/Users/arcticfox/parula_colors.npy')
    parulacmap = LinearSegmentedColormap.from_list('sequential',
                                                   np.load('/Users/arcticfox/parula_data.npy'))
    cinds = parula[np.linspace(0, 210, len(histplots), dtype=int)]
    oppo = ['#a79ae6', '#84b3e3', '#6cdefa', '#aae8a0', '#f5cd54']
    return parula, cinds, oppo
    

def plot_hr(stars, ruwe_cutoff=1.4, save=True, outputname='ruwe_hr.pdf',
            flaresvmin=0, flaresvmax=1, plot_ruwe=True):
    
    cmd_bprp, cmd_mg, cmd_mass, cmd_teff = read_mamajek()

    if plot_ruwe:
        fig, (ax2, ax1) = plt.subplots(nrows=2, figsize=(16,16), 
                                       sharex=True, sharey=True)

    fig.set_facecolor('w')

    cm = flarerate_cmap()
    cm1 = ruwe_cmap()
    
    absmag = stars['phot_g_mean_mag'] - 5*np.log10(stars['TICv8_d']/10)
    
    # Subplot by RUWE

    ax1.plot(stars['bp_rp'], absmag, '.', 
             c='#b3b3b3', ms=2, alpha=0.3, zorder=0)
    im = ax1.scatter(stars['bp_rp'][stars['RUWE']>=ruwe_cutoff], 
                     absmag[stars['RUWE']>=ruwe_cutoff], 
                     c=stars['RUWE'][stars['RUWE']>=ruwe_cutoff], 
                     s=5, 
                     vmin=ruwe_cutoff, vmax=5, zorder=3,
                     cmap=cm1.reversed())
    fig.colorbar(im, ax=ax1, label='RUWE')


    # Subplot by flare rate

    good_inds = stars['RUWE'] > ruwe_cutoff
    im = ax2.scatter(stars['bp_rp'],
                     absmag,
                     c=stars['N_flares_per_day'],
                     s=5, 
                     vmin=flaresvmin, 
                     vmax=flaresvmax,
                     cmap=cm.reversed(), zorder=3)
    fig.colorbar(im, ax=ax2, label='Flare Rate [day$^{-1}$]')

    # Set plot limits, ticks, and labels

    xticks = [-1,0,1,2,3,4,5]
    upxticks = np.array([cmd_mass[cmd_bprp>=0][0], 
                         cmd_mass[cmd_bprp>=2][0], 
                         cmd_mass[cmd_bprp>=4][0]])
    ax1.set_xlim(-1, 5)
    ax2.set_xlim(-1, 5)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)

    plt.ylim(17,-5)

    ax1.set_xlabel('Gaia B$_p$ - R$_p$')
    ax1.set_ylabel('Gaia M$_G$')
    ax2.set_ylabel('Gaia M$_G$')
    plt.subplots_adjust(hspace=0.1)
    ax1.set_rasterized(True)
    ax2.set_rasterized(True)
    
    if save:
        plt.savefig(outputname, dpi=250, rasterize=True, bbox_inches='tight')

    return absmag


def plot_slopes(histplots, lowlim, upplim, mcmc_fits, starbins,
                save=True, outputname='amp_rates.pdf'):
    
    parula, cinds, oppo = parula_cmap(histplots)

    fig, (ax1, ax2) = plt.subplots(figsize=(14,8), ncols=2,
                                   gridspec_kw={'width_ratios':[12,0.5]})
    
    fig.set_facecolor('w')
    cinds = parula[np.linspace(0, 210, len(histplots), dtype=int)]
    
    
    for i in range(len(histplots)):
        q = histplots[i][0] > 0
        
        ax1.plot(histplots[i][1][1:][q], histplots[i][0][q], 
                 c=cinds[i], lw=2)
        
        ax1.scatter(histplots[i][1][1:][q], 
                    y=histplots[i][0][q], 
                    marker='o', c=cinds[i],
                    s=100, edgecolor='k', 
                    label='{} $\pm$ {}'.format(np.round(mcmc_fits[i][0],3),
                                               np.round(mcmc_fits[i][1], 3)),
                    zorder=100)
        
        ax1.fill_between(histplots[i][1][1:][q],
                         y1=lowlim[i][q],
                         y2=upplim[i][q],
                         color=cinds[i], alpha=0.4, lw=0)
        

    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    ax1.set_xlabel('Flare Amplitude [%]')
    ax1.set_ylabel('Flare Rate [Day$^{-1}$]')
    
    # Heinous colorbar hack #
    ax2.set_xticks([])
    ax2.set_xlim(0,1)

    for i in range(len(cinds)):
        ax2.plot(0.5, i+0.5, 's', ms=87,
                 c=cinds[i])
    ax2.yaxis.tick_right()
    ax2.set_yticklabels(starbins)
    ax2.set_ylim(0,len(cinds))
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Mass [M$_\odot$]')
    
    plt.subplots_adjust(wspace = 0.03)

    ax1.set_rasterized(True)
    ax2.set_rasterized(True)
    
    if save:
        plt.savefig(outputname, rasterize=True, dpi=300,
                    bbox_inches='tight', pad_inches=0)
