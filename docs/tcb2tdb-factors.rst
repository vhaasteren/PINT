.. highlight:: shell

How PINT converts TCB and TDB parameters
----------------------------------------

PINT evaluates timing models in TDB and does not evaluate ``UNITS TCB``
models directly. The converter therefore maps a TCB parameterization onto
PINT's existing TDB forward model.

Coordinate epochs are converted with :class:`astropy.time.Time`, which uses
ERFA's IAU 2006 TCB/TDB realization, including the defining ``TDB0`` offset.
The implementation does not duplicate the IAU constants.

For intervals, define ``F = d(TDB)/d(TCB) = 1 - L_B`` and ``K = 1/F``::

    dt_tdb = F * dt_tcb

Since the definition of the second is changing, all parameters involved in the timing model
must also be transformed. In the simplest case, if a quantity x has dimensions of [T^n], it
will be transformed as::
    
    x_tdb = x_tcb / K^n

This rule applies to the majority of parameters.

However, there are some parameters in pulsar timing which appear in the timing model multiplied 
by some constants. Examples include

    1. DM appears as DMconst * DM
    2. A1 appears as A1 / c
    3. M2 appears as M2 * G / c^3

In these cases, PINT keeps the multiplication factor numerically constant during
TCB <-> TDB conversion and absorbs the conversion into the parameter. If a
parameter has such a factor, specify it using the ``tcb2tdb_scale_factor``
argument while constructing the `Parameter` object. For example, DM will have 
`tcb2tdb_scale_factor=DMconst`.

Note that the parameter multiplied by the constant in these cases has dimensions of the 
form [T^n]. In the above cases, the value of n is as follows.

    1. A1 has n = 1
    2. M2 has n = 1

In general, if a parameter x appears in the timing model as C*x and if C*x has dimensionality of
the form [T^n], the scaling should be done with the "effective dimensionality" n.

If a parameter doesn't have a dimensionality of [T^n], a general rule is to reorganize the 
factors in the equation such that each group has a dimensionality [T^n]. This is ALWAYS possible
because the timing model components produce either a delay ([T^1]) or a phase ([T^0]).

A useful trick is to express parameters in geometrized units, where everything
has dimensions of ``T^n``. This is only the default: component-owned conversion
metadata overrides it when the forward model fixes an independent variable.

Radio-frequency decision
~~~~~~~~~~~~~~~~~~~~~~~~

PINT supports ``DILATEFREQ N``. Its barycentric radio frequency applies the
Earth-motion Doppler correction but is not dilated by ``K`` during unit
conversion. The converter holds that frequency numerically fixed.

Consequently, deterministic dispersion delays scale by ``F``. For the DM
Taylor series, order ``q`` scales as::

    DM_tdb^(q) = K^(q-1) DM_tcb^(q)

Thus DM scales by ``F``, DM1 is unchanged, and DM2 scales by ``K``. This
order-aware rule is represented by component-owned metadata rather than a
global parameter-name table.

Explicit invariants
~~~~~~~~~~~~~~~~~~~

PX is left numerically unchanged because PINT does not implement a matching
TCB/TDB spatial-coordinate transformation. ``START`` and ``FINISH`` are
data-span selectors and are also unchanged. UTC interval selectors such as
DMX range boundaries remain UTC and are not coordinate epochs.

Exceptions to this are noise parameters. The TOA uncertainties are measured in the observatory 
timescale and are not converted into TCB or TDB before computing the likelihood function/
chi-squared. Hence, we don't convert the quantities that modify TOA uncertainties, namely EFACs and``
EQUADs. Since we are not converting TOA variances, it doesn't make sense to convert TOA covariances
either. Hence, ECORRs and red and DM noise parameters are not converted. This means that 
the noise parameters must ALWAYS be re-estimated after a TCB <-> TDB conversion.

FD and FDJUMP coefficients are time-valued amplitudes evaluated at PINT's
fixed radio frequency, so every coefficient scales by ``F``. No logarithmic
coefficient mixing is needed.

Unsupported active deterministic terms are left unchanged and reported. They
do not prevent supported parameters from being converted. A conversion is
covered by the no-refit accuracy contract only when its report is accepted.
