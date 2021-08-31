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

def plot_hr(stars, ruwe_cutoff=1.4, save=True, outputname='ruwe_hr.pdf'):
    
    cmd_bprp, cmd_mg, cmd_mass, cmd_teff = read_mamajek()

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
                     vmin=0, vmax=0.5,
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
        plt.savefig('ruwe_hr.pdf', dpi=250, rasterize=True, bbox_inches='tight')

    return absmag
