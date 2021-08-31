import numpy as np
from astropy import units
from astropy.table import Table


def read_mamajek():
    cmd = Table.read('cmd_conversion.txt', format='ascii')
    cmd = cmd[cmd['Bp-Rp']!='...']
    
    cmd_bprp = np.array([float(i) for i in cmd['Bp-Rp']])
    cmd_mg = np.array([float(i) for i in cmd['M_G']])
    cmd_mass = np.array([float(i) for i in cmd['Msun']])
    cmd_teff = np.array([float(i) for i in cmd['Teff']])

    return cmd_bprp, cmd_mg, cmd_mass, cmd_teff


def ruwe_to_distance(stars):
    sigmaAL = Table.read('gaia_sigmaAL.txt', format='csv')
    poly = np.polyfit(sigmaAL['Gmag'].data, sigmaAL['sigmaAL'].data, deg=5)

    interp_sigAL = np.poly1d(poly)
    deltheta = interp_sigAL(stars['M_G'].data.data) * np.sqrt(stars['RUWE'].data**2-1)
    deltaa = (stars['TICv8_d']*units.pc).to(units.kpc) * deltheta
    return deltaa


def dist_to_ms(stars, starbins):

    cmd_bprp, cmd_mg, cmd_mass, cmd_teff = read_mamajek()

    subbprp = np.array([])
    subg = np.array([])

    for i in range(len(starbins)-1):
        inds = np.where((cmd_mass>=starbins[i]) & (cmd_mass<starbins[i+1]))[0]
        
        dat = stars[(stars['bp_rp']>= cmd_bprp[inds][0]) &
                    (stars['bp_rp']<=  cmd_bprp[inds][-1]) &
                    (stars['M_G']>= cmd_mg[inds][0]) &
                    (stars['M_G']<=  cmd_mg[inds][-1]) &
                    (stars['M_G'] <= 15)]

        subbprp = np.append(subbprp, dat['bp_rp'].data)
        subg = np.append(subg, dat['M_G'].data)

    fit = np.polyfit(subbprp, subg, deg=2)
    model = np.poly1d(fit)

    return model
