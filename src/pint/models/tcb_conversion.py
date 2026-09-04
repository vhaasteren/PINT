"""TCB/TDB conversion consistent with PINT's TDB forward model.

Coordinate epochs use Astropy/ERFA's IAU 2006 realization. Radio frequencies
remain undilated, matching PINT's supported ``DILATEFREQ N`` behavior.
Unsupported active deterministic terms are reported but do not stop conversion.
"""

from dataclasses import dataclass

import erfa
import numpy as np
from loguru import logger as log

from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    floatParameter,
    maskParameter,
    prefixParameter,
)
from pint.models.timing_model import TimingModel
from pint.pulsar_mjd import time_from_longdouble

__all__ = [
    "TCB_TDB_F",
    "TCB_TDB_K",
    "IFTE_K",
    "TCBTDBConversionReport",
    "scale_parameter",
    "transform_mjd_parameter",
    "convert_tcb_tdb",
]

# PINT evaluates its forward model in IAU 2006 TDB. Obtain the defining rate
# from ERFA rather than maintaining another decimal copy here.
TCB_TDB_F = np.longdouble(1) - np.longdouble(erfa.ELB)
TCB_TDB_K = np.longdouble(1) / TCB_TDB_F

# Backwards-compatible public alias. This is now the IAU/ERFA rate, not the
# historical IFTE common-origin epoch map.
IFTE_K = TCB_TDB_K


@dataclass(frozen=True)
class TCBTDBConversionReport:
    """Summary of operations performed during TCB/TDB conversion."""

    source_units: str
    target_units: str
    convention: str
    converted: tuple[str, ...]
    invariant: tuple[str, ...]
    unsupported: tuple[str, ...]
    unaudited_components: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        """Whether the conversion satisfies the tested no-refit contract."""
        return not self.unsupported and not self.unaudited_components


def scale_parameter(model: TimingModel, param: str, n: int, backwards: bool) -> None:
    """Scale a parameter x by a power of the IAU TCB/TDB rate K.

        x_tdb = x_tcb * K**n

    The power n depends on the "effective dimensionality" of
    the parameter as it appears in the timing model. Some examples
    are given bellow:

        1. F0 has effective dimensionality of frequency and n = 1
        2. F1 has effective dimensionality of frequency^2 and n = 2
        3. A1 has effective dimensionality of time because it appears as
           A1/c in the timing model. Therefore, its n = -1
        4. Constant DM has an explicit n = -1 override because PINT keeps
           radio frequency fixed during conversion
        5. PBDOT is dimensionless and has n = 0. i.e., it is not scaled.

    Parameter
    ---------
    model : pint.models.timing_model.TimingModel
        The timing model
    param : str
        The parameter name to be converted
    n : int
        The power of TCB_TDB_K in the scaling factor
    backwards : bool
        Whether to do TDB to TCB conversion.
    """
    assert isinstance(n, int), "The power must be an integer."

    p = -1 if backwards else 1

    factor = TCB_TDB_K ** (p * n)

    if (param in model) and model[param].quantity is not None:
        par = model[param]
        par.value *= factor
        if par.uncertainty_value is not None:
            par.uncertainty_value *= factor


def transform_mjd_parameter(model: TimingModel, param: str, backwards: bool) -> None:
    """Convert a coordinate epoch between TCB and IAU 2006 TDB.

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        The timing model
    param : str
        The parameter name to be converted
    backwards : bool
        Whether to do TDB to TCB conversion.
    """
    if (param in model) and model[param].quantity is not None:
        par = model[param]
        assert isinstance(par, MJDParameter) or (
            isinstance(par, prefixParameter)
            and isinstance(par.param_comp, MJDParameter)
        )

        source = "tdb" if backwards else "tcb"
        target = "tcb" if backwards else "tdb"
        converted = getattr(time_from_longdouble(par.value, source), target)

        # Re-label before assigning the already-converted two-part Time so the
        # old numeric MJD is not silently reinterpreted in the target scale.
        par.time_scale = target
        par.quantity = converted
        if par.uncertainty_value is not None:
            par.uncertainty_value *= TCB_TDB_K if backwards else TCB_TDB_F


def _scale_exponent(param) -> int:
    """Return the conversion exponent, honoring component-owned overrides."""
    override = getattr(param, "tcb2tdb_scale_exponent", None)
    if callable(override):
        override = override(param)
    if override is not None:
        return int(override)
    return -param.effective_dimensionality


def _parameter_conversion_plan(
    model: TimingModel,
) -> tuple[set[str], set[str], set[str]]:
    """Classify set parameters without changing the model."""
    convertible: set[str] = set()
    invariant: set[str] = set()
    unsupported: set[str] = set()

    for name in model.params:
        param = model[name]
        if param.quantity is None:
            continue
        if getattr(param, "tcb2tdb_invariant", False):
            invariant.add(name)
            continue
        if not getattr(param, "convert_tcb2tdb", False):
            continue
        if isinstance(param, (floatParameter, AngleParameter, maskParameter)) or (
            isinstance(param, prefixParameter)
            and isinstance(param.param_comp, (floatParameter, AngleParameter))
        ):
            if _scale_exponent(param) == 0:
                invariant.add(name)
            else:
                convertible.add(name)
        elif isinstance(param, MJDParameter) or (
            isinstance(param, prefixParameter)
            and isinstance(param.param_comp, MJDParameter)
        ):
            if param.time_scale == "utc":
                invariant.add(name)
            elif param.time_scale in {"tcb", "tdb"}:
                convertible.add(name)
            else:
                unsupported.add(name)
        else:
            unsupported.add(name)

    return convertible, invariant, unsupported


def _active_unsupported_components(
    model: TimingModel, converted: set[str], invariant: set[str]
) -> tuple[set[str], set[str]]:
    """Find unhandled parameters from PINT's actual delay/phase graph."""
    unsupported: set[str] = set()
    components: set[str] = set()
    forward_components = model.DelayComponent_list + model.PhaseComponent_list

    for component in forward_components:
        component_unsupported = set()
        for name in component.params:
            par = model[name]
            if par.quantity is None or name in converted or name in invariant:
                continue
            if hasattr(par, "convert_tcb2tdb"):
                component_unsupported.add(name)
        if component_unsupported:
            unsupported.update(component_unsupported)
        if component_unsupported or not component.tcb2tdb_certified:
            components.add(component.__class__.__name__)

    return unsupported, components


def convert_tcb_tdb(
    model: TimingModel, backwards: bool = False
) -> TCBTDBConversionReport:
    """Convert every supported parameter between TCB and TDB.

    The conversion follows PINT's IAU 2006 TDB, undilated-radio-frequency
    forward model. Unsupported active deterministic terms are left unchanged
    and reported; they never prevent supported operations from completing.

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
       Timing model to be converted.
    backwards : bool
        Whether to do TDB to TCB conversion. The default is TCB to TDB.

    Returns
    -------
    TCBTDBConversionReport
        Actual converted, invariant, and unsupported model terms. An accepted
        report is covered by PINT's tested no-refit conversion contract.
    """

    target_units = "TCB" if backwards else "TDB"
    source_units = "TDB" if backwards else "TCB"
    convertible, invariant, unsupported = _parameter_conversion_plan(model)

    if model["UNITS"].value == target_units or (
        model["UNITS"].value is None and not backwards
    ):
        log.warning("The input par file is already in the target units. Doing nothing.")
        graph_unsupported, unaudited_components = _active_unsupported_components(
            model, convertible, invariant
        )
        unsupported.update(graph_unsupported)
        report = TCBTDBConversionReport(
            source_units=target_units,
            target_units=target_units,
            convention="iau2006-undilated-frequency",
            converted=(),
            invariant=tuple(sorted(invariant)),
            unsupported=tuple(sorted(unsupported)),
            unaudited_components=tuple(sorted(unaudited_components)),
        )
        model.tcb_tdb_conversion_report = report
        return report

    for name in convertible:
        param = model[name]
        if isinstance(param, (floatParameter, AngleParameter, maskParameter)) or (
            isinstance(param, prefixParameter)
            and isinstance(param.param_comp, (floatParameter, AngleParameter))
        ):
            scale_parameter(model, name, _scale_exponent(param), backwards)
        else:
            transform_mjd_parameter(model, name, backwards)

    graph_unsupported, unaudited_components = _active_unsupported_components(
        model, convertible, invariant
    )
    unsupported.update(graph_unsupported)

    model["UNITS"].value = target_units

    model.validate(allow_tcb=backwards)

    report = TCBTDBConversionReport(
        source_units=source_units,
        target_units=target_units,
        convention="iau2006-undilated-frequency",
        converted=tuple(sorted(convertible)),
        invariant=tuple(sorted(invariant)),
        unsupported=tuple(sorted(unsupported)),
        unaudited_components=tuple(sorted(unaudited_components)),
    )
    model.tcb_tdb_conversion_report = report

    if not report.accepted:
        log.warning(
            "TCB/TDB conversion completed, but the no-refit accuracy contract "
            "does not cover this model. Unsupported parameters: {}. "
            "Unaudited components: {}.",
            ", ".join(report.unsupported) or "none",
            ", ".join(report.unaudited_components) or "none",
        )

    return report
