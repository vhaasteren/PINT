"""Tests for `pint.models.tcb_conversion` and the `tcb2tdb` script."""

import os
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
import astropy.units as u

from pint import DMconst, dmu
from pint.models.model_builder import ModelBuilder, get_model
from pint.models.tcb_conversion import TCB_TDB_F, TCB_TDB_K, convert_tcb_tdb
from pint.pulsar_mjd import time_from_longdouble
from pint.scripts import tcb2tdb

simplepar = """
PSR              PSRTEST
RAJ       17:48:52.75  1
DECJ      -20:21:29.0  1
F0       61.485476554  1
F1         -1.181D-15  1
PEPOCH        53750.000000
POSEPOCH      53750.000000
DM              223.9  1
SOLARN0               0.00
BINARY              BT
T0                  53750
A1                  100.0 1 0.1
ECC                 1.0
OM                  0.0
PB                  10.0
FD1                 1e-3
EPHEM               DE436
CLK              TT(BIPM2017)
UNITS               TCB
TIMEEPH             FB90
T2CMETHOD           TEMPO
CORRECT_TROPOSPHERE N
PLANET_SHAPIRO      Y
DILATEFREQ          N
"""


@pytest.mark.parametrize("backwards", [True, False])
def test_convert_units(backwards):
    with pytest.raises(ValueError):
        m = ModelBuilder()(StringIO(simplepar))

    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    f0_tcb = m.F0.value
    pb_tcb = m.PB.value
    convert_tcb_tdb(m, backwards=backwards)
    assert m.UNITS.value == ("TCB" if backwards else "TDB")
    assert np.isclose(m.F0.value / f0_tcb, pb_tcb / m.PB.value)


def test_convert_units_roundtrip():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    m_ = deepcopy(m)
    convert_tcb_tdb(m, backwards=False)
    convert_tcb_tdb(m, backwards=True)

    for par in m.params:
        p = getattr(m, par)
        p_ = getattr(m_, par)
        if p.value is None:
            assert p_.value is None
        elif isinstance(p.value, str):
            assert getattr(m, par).value == getattr(m_, par).value
        else:
            assert np.isclose(getattr(m, par).value, getattr(m_, par).value)


def test_coordinate_epoch_uses_astropy_iau_tdb():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    original = time_from_longdouble(m.PEPOCH.value, "tcb")
    expected = original.tdb
    m.PEPOCH.uncertainty_value = 1

    report = convert_tcb_tdb(m)

    assert abs((m.PEPOCH.quantity - expected).to_value(u.ns)) < 0.01
    assert np.isclose(m.PEPOCH.uncertainty_value, TCB_TDB_F)
    assert m.PEPOCH.time_scale == "tdb"
    assert "PEPOCH" in report.converted

    convert_tcb_tdb(m, backwards=True)
    assert abs((m.PEPOCH.quantity - original).to_value(u.ns)) < 0.01
    assert m.PEPOCH.time_scale == "tcb"


def test_coordinate_epoch_includes_tdb0():
    reference_mjd = np.longdouble("43144.0003725")
    converted = time_from_longdouble(reference_mjd, "tcb").tdb
    numeric_offset = (converted.mjd_long - reference_mjd) * u.day

    assert np.isclose(numeric_offset.to_value(u.us), -65.5, atol=0.01)


def test_fixed_frequency_dm_and_fd_scaling():
    m = ModelBuilder()(
        StringIO(
            simplepar.replace(
                "DM              223.9  1",
                "DM              223.9  1\nDM1 1e-3\nDM2 2e-5\nDMEPOCH 53750",
            )
        ),
        allow_tcb="raw",
    )
    dm0, dm1, dm2 = m.DM.value, m.DM1.value, m.DM2.value
    frequencies = np.array([0.7, 1.4, 3.0]) * u.GHz
    fd_delay = m.FD_delay_frequency(frequencies)
    tcb_mjds = np.array([53750.0, 54000.0, 55000.0], dtype=np.longdouble)
    tdb_mjds = np.array(
        [time_from_longdouble(t, "tcb").tdb.mjd_long for t in tcb_mjds],
        dtype=np.longdouble,
    )
    dm_component = m.components["DispersionDM"]
    dm_tcb = dm_component.base_dm({"tdbld": tcb_mjds})
    dm_delay_tcb = dm_component.dispersion_time_delay(dm_tcb, frequencies)

    convert_tcb_tdb(m)

    assert np.isclose(m.DM.value / dm0, TCB_TDB_F)
    assert np.isclose(m.DM1.value / dm1, 1)
    assert np.isclose(m.DM2.value / dm2, TCB_TDB_K)
    fd_error = m.FD_delay_frequency(frequencies) - fd_delay * TCB_TDB_F
    assert np.max(np.abs(fd_error.to_value(u.ns))) < 0.01

    dm_tdb = dm_component.base_dm({"tdbld": tdb_mjds})
    dm_delay_tdb = dm_component.dispersion_time_delay(dm_tdb, frequencies)
    dm_error = dm_delay_tdb - dm_delay_tcb * TCB_TDB_F
    assert np.max(np.abs(dm_error.to_value(u.ns))) < 0.01


def test_spindown_phase_closes_without_refitting():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    tcb_mjds = np.array([53750.0, 54000.0, 55000.0], dtype=np.longdouble)
    tdb_mjds = np.array(
        [time_from_longdouble(t, "tcb").tdb.mjd_long for t in tcb_mjds],
        dtype=np.longdouble,
    )
    zero_delay = np.zeros(len(tcb_mjds)) * u.s
    phase_tcb = m.components["Spindown"].spindown_phase(
        SimpleNamespace(table={"tdbld": tcb_mjds}), zero_delay
    )

    convert_tcb_tdb(m)

    phase_tdb = m.components["Spindown"].spindown_phase(
        SimpleNamespace(table={"tdbld": tdb_mjds}), zero_delay
    )
    time_error = (phase_tdb - phase_tcb) / m.F0.quantity
    assert np.max(np.abs(time_error.to_value(u.ns))) < 0.01


def test_astrometric_position_closes_at_same_physical_epoch():
    m = ModelBuilder()(
        StringIO(
            """
PSR TEST
RAJ 12:00:00
DECJ 20:00:00
PMRA 8
PMDEC -5
POSEPOCH 55000
F0 100
PEPOCH 55000
DM 10
UNITS TCB
"""
        ),
        allow_tcb="raw",
    )
    # A raw TCB model is intentionally not evaluable by PINT, so label the
    # source epoch explicitly for this component-level physical-instant test.
    m.POSEPOCH.time_scale = "tcb"
    epoch_tcb = time_from_longdouble(np.longdouble("57000"), "tcb")
    position_tcb = m.coords_as_ICRS(epoch=epoch_tcb)

    convert_tcb_tdb(m)

    position_tdb = m.coords_as_ICRS(epoch=epoch_tcb.tdb)
    assert position_tdb.separation(position_tcb).to_value(u.uas) < 1e-3


def test_fdjump_coefficients_scale_as_time_at_fixed_frequency():
    m = ModelBuilder()(
        StringIO(simplepar + "\nFD1JUMP -sys backend 0.01\n"),
        allow_tcb="raw",
    )
    original = m.FD1JUMP1.value

    convert_tcb_tdb(m)

    assert np.isclose(m.FD1JUMP1.value / original, TCB_TDB_F)


def test_utc_selectors_and_parallax_are_explicit_invariants():
    m = ModelBuilder()(
        StringIO(
            simplepar
            + """
PX 1.2
START 53000
FINISH 54000
DMXR1_0001 53500
DMXR2_0001 53600
DMX_0001 0.1
"""
        ),
        allow_tcb="raw",
    )
    values = {
        name: m[name].value
        for name in ("PX", "START", "FINISH", "DMXR1_0001", "DMXR2_0001")
    }

    report = convert_tcb_tdb(m)

    assert {name: m[name].value for name in values} == values
    assert set(values) <= set(report.invariant)
    assert np.isclose(m.DMX_0001.value, 0.1 * TCB_TDB_F)


def test_report_accepts_only_audited_forward_components():
    m = ModelBuilder()(
        StringIO(
            """
PSR TEST
RAJ 12:00:00
DECJ 20:00:00
F0 100
PEPOCH 55000
DM 12.5
UNITS TCB
"""
        ),
        allow_tcb="raw",
    )
    assert convert_tcb_tdb(m).accepted

    binary = ModelBuilder()(StringIO(simplepar), allow_tcb="raw")
    report = convert_tcb_tdb(binary)
    assert not report.accepted
    assert "BinaryBT" in report.unaudited_components


def test_effective_dimensionality():
    m = ModelBuilder()(StringIO(simplepar), allow_tcb=True)
    assert m.PEPOCH.effective_dimensionality == 1
    assert m.F0.effective_dimensionality == -1
    assert m.F1.effective_dimensionality == -2

    assert m.POSEPOCH.effective_dimensionality == 1
    assert m.RAJ.effective_dimensionality == 0
    assert m.DECJ.effective_dimensionality == 0
    assert m.PMRA.effective_dimensionality == -1
    assert m.PMDEC.effective_dimensionality == -1
    assert m.PX.effective_dimensionality == -1

    assert m.DM.effective_dimensionality == -1

    assert m.T0.effective_dimensionality == 1
    assert m.A1.effective_dimensionality == 1
    assert m.ECC.effective_dimensionality == 0
    assert m.OM.effective_dimensionality == 0
    assert m.PB.effective_dimensionality == 1

    assert m.NE_SW.effective_dimensionality == -2


def test_dm_scaling_factor():
    m = get_model(
        StringIO(
            """
            PSR         TEST
            F0          100
            F1          -1e-14
            PEPOCH      55000
            DMEPOCH     55000
            DM          12.5
            DM1         -0.001
            DM2         1e-5
            DMXR1_0001  51000
            DMXR2_0001  51000
            DMX_0001    0.002
            DMWXEPOCH   55000
            DMWXFREQ_0001   0.001
            DMWXSIN_0001    0.0003
            DMWXCOS_0001    0.0002
            """
        )
    )

    for param in m.params:
        par = m[param]

        if hasattr(par, "units") and par.units == dmu:
            assert not par.convert_tcb2tdb or par.tcb2tdb_scale_factor == DMconst


def test_tcb2tdb(tmp_path):
    tmppar1 = tmp_path / "tmp1.par"
    tmppar2 = tmp_path / "tmp2.par"
    with open(tmppar1, "w") as f:
        f.write(simplepar)

    cmd = f"{tmppar1} {tmppar2}"
    tcb2tdb.main(cmd.split())

    assert os.path.isfile(tmppar2)

    m2 = ModelBuilder()(tmppar2)
    assert m2.UNITS.value == "TDB"
