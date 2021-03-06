import utils
import numpy as np
import astropy.units as u
try:
    import astropy.erfa as erfa
except ImportError:
    import astropy._erfa as erfa
import astropy.table as table
from . import observatories as obsmod

SECS_PER_DAY = erfa.DAYSEC

from astropy.utils.iers import IERS_A, IERS_A_URL, IERS_B, IERS_B_URL, IERS
from astropy.utils.data import download_file
# iers_a_file = download_file(IERS_A_URL, cache=True)
iers_b_file = download_file(IERS_B_URL, cache=True)
# iers_a = IERS_A.open(iers_a_file)
iers_b = IERS_B.open(iers_b_file)
IERS.iers_table = iers_b
iers_tab = IERS.iers_table

# Read in the observatory information
observatories = obsmod.read_observatories()

# Earth rotation rate in radians per UT1 second
#
# This is from Capitaine, Guinot, McCarthy, 2000 and is
# in IAU Resolution B1.8 on the Earth Rotation Angle (ERA)
# and the relation of it to UT1.  The number 1.00273781191135448
# below is a defining constant.  See here:
# http://iau-c31.nict.go.jp/pdf/2009_IAUGA_JD6/JD06_capitaine_wallace.pdf
OM = 1.00273781191135448 * 2.0 * np.pi / SECS_PER_DAY

# arcsec to radians
asec2rad = 4.84813681109536e-06

def topo_posvels(obsname, toas):
    """Return a list of PosVel instances for the observatory at the TOA times

    This routine returns a list of PosVel instances, containing the
    positions (m) and velocities (m / s) at the times of the toas and
    referenced to the ITRF geocentric coordinates.  This routine is
    basically SOFA's pvtob() with an extra rotation from c2ixys.
    """
    # If the input is a single TOA (i.e. a row from the table),
    # then put it into a list
    if type(toas) == table.row.Row:
        ttoas = [toas['mjd']]
        N = 1
    else:
        N = len(toas)
        ttoas = toas['mjd']

    # Get various times from the TOAs as arrays
    tts = np.asarray([(toa.tt.jd1, toa.tt.jd2) for toa in ttoas]).T
    ut1s = np.asarray([(toa.ut1.jd1, toa.ut1.jd2) for toa in ttoas]).T
    mjds = np.asarray([toa.mjd for toa in ttoas])

    # Get x, y coords of Celestial Intermediate Pole and CIO locator s
    X, Y, S = erfa.xys00a(*tts)

    # Get dX and dY from IERS A in arcsec and convert to radians
    #dX = np.interp(mjds, iers_tab['MJD'], iers_tab['dX_2000A_B']) * asec2rad
    #dY = np.interp(mjds, iers_tab['MJD'], iers_tab['dY_2000A_B']) * asec2rad
    # Get dX and dY from IERS B in arcsec and convert to radians
    dX = np.interp(mjds, iers_tab['MJD'], iers_tab['dX_2000A']) * asec2rad
    dY = np.interp(mjds, iers_tab['MJD'], iers_tab['dY_2000A']) * asec2rad

    # Get GCRS to CIRS matrices
    rc2i = erfa.c2ixys(X+dX, Y+dY, S)

    # Gets the TIO locator s'
    sp = erfa.sp00(*tts)

    # Get X and Y from IERS A in arcsec and convert to radians
    #xp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_X_B']) * asec2rad
    #yp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_Y_B']) * asec2rad
    # Get X and Y from IERS B in arcsec and convert to radians
    xp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_x']) * asec2rad
    yp = np.interp(mjds, iers_tab['MJD'], iers_tab['PM_y']) * asec2rad

    # Get the polar motion matrices
    rpm = erfa.pom00(xp, yp, sp)

    # Observatory geocentric coords in m
    xyzm = np.array([a.to(u.m).value for a in \
                     observatories[obsname].loc.geocentric])
    x, y, z = np.dot(xyzm, rpm).T

    # Functions of Earth Rotation Angle
    theta = erfa.era00(*ut1s)
    s, c = np.sin(theta), np.cos(theta)
    sx, cx = s * x, c * x
    sy, cy = s * y, c * y

    # Initial positions and velocities
    iposs = np.asarray([cx - sy, sx + cy, z]).T
    ivels = np.asarray([OM * (-sx - cy), OM * (cx - sy), \
                        np.zeros_like(x)]).T
    # There is probably a way to do this with np.einsum or something...
    poss = np.empty((N, 3), dtype=np.float64)
    vels = np.empty((N, 3), dtype=np.float64)
    for ii in range(N):
        poss[ii] = np.dot(iposs[ii], rc2i[ii])
        vels[ii] = np.dot(ivels[ii], rc2i[ii])

    # Make a PosVel list for return
    pvs = [utils.PosVel(p, v, obj=obsname, origin="EARTH") \
           for p, v in zip(poss * u.m, vels * u.m / u.s)]
    return pvs




