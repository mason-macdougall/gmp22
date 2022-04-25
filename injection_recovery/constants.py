import numpy as np
import astropy.constants as apc
import uncertainties
from uncertainties import ufloat

### General Constants

pi = np.pi
RSAU = apc.R_sun.value/apc.au.value

G = 6.6743 * 10**(-8)            # cm^3 / (g * s^2)
G_err = 1.5 * 10**(-12)

msun = 1.988409870698051*10**33           # g
msun_err = 4.468805426856864*10**28
mearth = 5.9722*10**27

rsun = 6.957 * 10**10           # cm
rsun_err = 1.4 * 10**7

rearth = 6.378137 * 10**8         # cm
rearth_err = 1.0 * 10**4

G_u = ufloat(G, G_err)
msun_u = ufloat(msun, msun_err)
rsun_u = ufloat(rsun, rsun_err)
rearth_u = ufloat(rearth, rearth_err)

rhosun_u = msun_u / (4/3*np.pi * rsun_u**3)
rhosun = rhosun_u.n
rhosun_err = rhosun_u.s           # g / cc

day = 86400.                       # seconds

pi_2 = np.pi/2.0
pi_180 = np.pi/180.0



BIGG = apc.G.value

MSUN = apc.M_sun.value
RSUN = apc.R_sun.value

MEARTH = apc.M_earth.value
REARTH = apc.R_earth.value

MJUP = apc.M_jup.value
RJUP = apc.R_jup.value

MSME = MSUN/MEARTH
RSRE = RSUN/REARTH

