from pint import toa, utils, erfautils
import pint.observatories as obsmod
import math, shlex, subprocess, numpy
import astropy.constants as const
import astropy.units as u
from pint.utils import PosVel
from astropy import log
import os

testdir=os.path.join(os.getenv('PINT'),'tests');
datadir = os.path.join(testdir,'datafile')

log.setLevel('ERROR')
# for nice output info, set the following instead
#log.setLevel('INFO')

observatories = obsmod.read_observatories()

ls = u.def_unit('ls', const.c * 1.0 * u.s)

log.info("Reading TOAs into PINT")
ts = toa.get_TOAs(datadir + "/testtimes.tim",usepickle=False)
if log.level < 25:
    ts.print_summary()
ts.table.sort('index')

log.info("Calling TEMPO2")
#cmd = 'tempo2 -output general2 -f tests/testtimes.par tests/testtimes.tim -s "XXX {clock0} {clock1} {clock2} {clock3} {tt} {t2tb} {telSSB} {telVel} {Ttt}\n"'
cmd = 'tempo2 -output general2 -f ' + datadir+'/testtimes.par ' + datadir + \
      '/testtimes.tim -s "XXX {clock0} {clock1} {clock2} {clock3} {tt} {t2tb} {earth_ssb1} {earth_ssb2} {earth_ssb3} {earth_ssb4} {earth_ssb5} {earth_ssb6} {telEpos} {telEVel} {Ttt}\n"'
args = shlex.split(cmd)

tout = subprocess.check_output(args)
goodlines = [x for x in tout.split("\n") if x.startswith("XXX")]

assert(len(goodlines)==len(ts.table))

# Get the output lines from the TOAs
for line, TOA in zip(goodlines, ts.table):
    assert len(line.split()) == 20, \
      "tempo2 general2 does not support all needed outputs"
    oclk, ut1_utc, tai_utc, tt_tai, ttcorr, tt2tb, \
          ep0, ep1, ep2, ev0, ev1, ev2, \
          tp0, tp1, tp2, tv0, tv1, tv2, Ttt = \
          (float(x) for x in line.split()[1:])
    t2_epv = utils.PosVel(numpy.asarray([ep0, ep1, ep2]) * ls,
                          numpy.asarray([ev0, ev1, ev2]) * ls/u.s)
    t2_opv = utils.PosVel(numpy.asarray([tp0, tp1, tp2]) * ls,
                          numpy.asarray([tv0, tv1, tv2]) * ls/u.s)

    t2_ssb2obs = t2_epv + t2_opv
    # print utils.time_toq_mjd_string(TOA.mjd.tt), line.split()[-1]
    tempo_tt = utils.time_from_mjd_string(line.split()[-1], scale='tt')
    # Ensure that the clock corrections are accurate to better than 0.1 ns
    assert(math.fabs((oclk*u.s - TOA['flags']["clkcorr"]).to(u.ns).value) < 0.1)

    log.info("TOA in tt difference is: %.2f ns" % \
             ((TOA['mjd'].tt - tempo_tt.tt).sec * u.s).to(u.ns).value)

    pint_opv = erfautils.topo_posvels(TOA['obs'], TOA)[0] # usually for arrays...
    #print " obs  T2:", t2_opv.pos.to(u.m).value, t2_opv.vel.to(u.m/u.s)
    #print " obs PINT:", pint_opv.pos.to(u.m), pint_opv.vel.to(u.m/u.s)
    dopv = pint_opv - t2_opv
    dpos = numpy.sqrt(numpy.dot(dopv.pos.to(u.m), dopv.pos.to(u.m)))
    dvel = numpy.sqrt(numpy.dot(dopv.vel.to(u.mm/u.s), dopv.vel.to(u.mm/u.s)))
    log.info(" obs diff: %.2f m, %.3f mm/s" % (dpos, dvel))
    assert(dpos < 2.0 and dvel < 0.02)

    pint_ssb2obs = PosVel(numpy.asarray(TOA['ssb_obs_pos'])*u.km,
                          numpy.asarray(TOA['ssb_obs_vel'])*u.km/u.s,
                          origin='SSB', obj='OBS')
    #print " topo  T2:", t2_ssb2obs.pos.to(u.km), t2_ssb2obs.vel.to(u.km/u.s)
    #print " topo PINT:", pint_ssb2obs.pos.to(u.km), pint_ssb2obs.vel.to(u.km/u.s)
    dtopo = pint_ssb2obs - t2_ssb2obs
    dpos = numpy.sqrt(numpy.dot(dtopo.pos.to(u.m), dtopo.pos.to(u.m)))
    dvel = numpy.sqrt(numpy.dot(dtopo.vel.to(u.mm/u.s), dtopo.vel.to(u.mm/u.s)))
    log.info(" topo diff: %.2f m, %.3f m/s" % (dpos, dvel))
    assert(dpos < 2.0 and dvel < 0.02)
