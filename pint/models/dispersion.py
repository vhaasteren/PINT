"""This module implements a simple model of a constant dispersion measure."""
# dispersion.py
# Simple (constant) ISM dispersion measure
from warnings import warn
from .parameter import Parameter, DMXParameter
from .timing_model import TimingModel, Cache
import astropy.units as u
from astropy import log
import numpy as np

# The units on this are not completely correct
# as we don't really use the "pc cm^3" units on DM.
# But the time and freq portions are correct
DMconst = 1.0/2.41e-4 * u.MHz * u.MHz * u.s

class SimpleDispersion(TimingModel):
    """This class provides a timing model for a simple constant
    dispersion measure.
    """

    def __init__(self):
        super(Dispersion, self).__init__()

        self.add_param(Parameter(name="DM",
            units="pc cm^-3", value=0.0,
            description="Dispersion measure"))
        self.delay_funcs += [self.dispersion_delay,]

    def setup(self):
        super(Dispersion, self).setup()

    @Cache.use_cache
    def dispersion_delay(self, toas):
        """Return the dispersion delay at each toa."""
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']
        return self.DM.value * DMconst / bfreq**2

    @Cache.use_cache
    def d_delay_d_DM(self, toas):
        """Return the dispersion delay at each toa."""
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']
        return DMconst / bfreq**2

class Dispersion(TimingModel):
    """This class provides a timing model for a simple constant
    dispersion measure.
    """

    def __init__(self):
        super(Dispersion, self).__init__()

        self.add_param(Parameter(name="DM",
            units="pc cm^-3", value=0.0,
            description="Dispersion measure"))

        # These functions actually add index 0001 (arg is the 'current' par)
        self.add_DMX_parameter('DMX_0000')
        self.add_DMX_parameter('DMXR1_0000')
        self.add_DMX_parameter('DMXR2_0000')

        self.delay_funcs += [self.dispersion_delay,]

    def setup(self):
        super(Dispersion, self).setup()

    @Cache.use_cache
    def get_dmx_mapping(self, prefix):
        """Obtain the index to parameter-name mapping. Necessary, since we don't
        know how many leading zero's are in the parfile"""
        parnames = [x for x in self.params if x.startswith(prefix)]
        mapping = dict()
        for parname in parnames:
            index = int(parname[len(prefix):])
            mapping[index] = parname

        return mapping

    @Cache.use_cache
    def dispersion_delay(self, toas):
        """Return the dispersion delay at each toa."""
        # TODO: DMX range check not against BATs, but against SATs
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']

        # Constant DM delay
        dmdelay = self.DM.value * DMconst / bfreq**2

        # DMX delays
        dmx_mapping = self.get_dmx_mapping('DMX_')
        dmxr1_mapping = self.get_dmx_mapping('DMXR1_')
        dmxr2_mapping = self.get_dmx_mapping('DMXR2_')
        epoch_ind = 1
        while epoch_ind in dmx_mapping:
            # Get the parameters
            r1 = getattr(self, dmxr1_mapping[epoch_ind]).value
            r2 = getattr(self, dmxr2_mapping[epoch_ind]).value
            dmx = getattr(self, dmx_mapping[epoch_ind]).value

            # Apply the DMX delays
            msk = np.logical_and(toas['tdbld'] >= r1, toas['tdbld'] <= r2)
            dmdelay[msk] += dmx * DMconst / bfreq[msk]**2

            epoch_ind = epoch_ind + 1

        return dmdelay

    def add_DMX_parameter(self, cur_par):
        """Add a DMX_, DMXR1_, or DMXR2_ parameter"""
        prefixes = ['DMX_', 'DMXR1_', 'DMXR2_']
        dmxunits = {'DMX_':'pc cm^-3',
                    'DMXR1_':'MJD',
                    'DMXR2_':'MJD'}
        dmxdescription = {'DMX_':'Dispersion measure variation',
                          'DMXR1_':'Beginning of DMX interval',
                          'DMXR2_':'End of DMX interval'}

        for prefix in prefixes:
            lp = len(prefix)
            if cur_par[:lp] == prefix and cur_par[lp:].isdigit():
                digit, digitlen = int(cur_par[lp:]), len(cur_par[lp:])
                newdigit = digit+1
                parformat = prefix + '{0:0'+str(digitlen)+'d}'
                newparname = parformat.format(newdigit)
                self.add_param(DMXParameter(name=newparname,
                        units=dmxunits[prefix], value=0.0,
                        description=dmxdescription[prefix],
                        add_par_callback=self.add_DMX_parameter,
                        prefix=prefix))

    @Cache.use_cache
    def d_delay_d_DM(self, toas):
        """Return the dispersion delay at each toa."""
        try:
            bfreq = self.barycentric_radio_freq(toas)
        except AttributeError:
            warn("Using topocentric frequency for dedispersion!")
            bfreq = toas['freq']
        return DMconst / bfreq**2
