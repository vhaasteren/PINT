# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Every active FBX model now uses FB coefficients as its complete orbital-phase representation. An ordinary `PBDOT` supplied with `FB0` is converted to `FB1` instead of being silently ignored. `PB` and applicable `PBDOT` values are exposed as derived views, so `as_parfile()` gains commented `# PB`/`# PBDOT` lines for existing pure-FBX models and no longer exposes an unset writable `PBDOT` that the FBX orbit would ignore.
- DDGR+FBX is now rejected explicitly before normalization. The new general PB/FB bridge does not broaden DDGR into a partially supported configuration or rely on incidental missing-FB0/PB-conflict failures. Correct support requires dynamic synchronization of DDGR's PB-dependent post-Keplerian quantities and their analytic `PB(FB0)` chain-rule derivatives.
- `BinaryDDH` now accepts a negative fitted `H3` with a warning when `STIGMA` is finite and positive. The DDH delay is smooth in the signed amplitude `H3`, but the resulting negative derived `M2` is not physically interpretable. Validation of free `M2`, generic `SINI`, and non-DDH binary models is unchanged.
- `d_delay_d_param` now applies the chain rule for delay components that respond to delays accumulated from earlier components (e.g. a binary delay's dependence on its evaluation epoch). Design-matrix entries for binary pulsars change at the ~1e-5 relative level. The parameter-independent ingredients are computed once per design matrix via the new `TimingModel.delay_deriv_chain`; the unused `acc_delay` argument of `d_delay_d_param` was replaced by the optional `chain` argument.
### Added
- Support for hierarchical triple systems: a second (outer) binary component can be added via a `BINARY2` line with `_2`-suffixed orbital parameters (e.g. `PB_2`, `A1_2`). Outer orbit delay is computed before, and propagated into, the inner binary. Outer wrappers `BinaryDD2`, `BinaryBT2`, and `BinaryELL12` are provided. Delay derivatives account for the outer→inner coupling (chain rule through the previous delay). The projected semi-major axis includes a second time derivative `A1DOT2` (alias `X2DOT`).
- Time-domain solar wind GP noise components: ridge, squared-exponential, Matérn, and quasi-periodic kernels
- Documentation page explaining the time-domain solar wind noise model, its interpolation basis, and how it differs from the Fourier-basis noise models
- `TOAs.get_tdb_seconds()`, returning the TDB times of the TOAs in seconds with a selectable dtype
- `BINARY DDR` (Damour-Deruelle-Regular binary): `BinaryDDR` wrap around the stand-alone delay kernel; same `PulsarBinary` `FB0`-`FBn` tokens as DD/ELL1, with the kernel evaluating the full prefix (scaled Horner, centering, `DDR_FB_KMAX`; no mixed `PBDOT` / `DDRKINE Y`); analytic astrometric and batched full-model derivatives including the `-B_t A_θ` time-argument correction, strict unsupported-parameter validation, WLS recovery, and exact secular/gauge-aware `convert_binary` import/export with fit-space and uncertainty reporting. ELL1 `TASC` is translated by \(3/2\,xh\) with the phase chart and `A1` copied; ELL1H absorbed conversions apply the Freire-Wex orbit encode/decode once as the complete map.
- Binary orbital-phase normalization now accepts `PB` (and optionally ordinary `PBDOT`) together with `FBn` while `FB0` is omitted. PINT converts the nominal model to a canonical FBX Taylor series, inserts frozen zero coefficients for sparse indices, and exposes `PB`/`PBDOT` as read-only derived views. Every valued FB coefficient is now guaranteed to contribute to orbital phase or produce a clear error.
### Fixed
- Sparse FBX series no longer silently ignore coefficients after the first missing index.
- Propagate one-way astrometric marginal uncertainties in ``as_ECL`` / ``as_ICRS`` by diagonal covariance rotation instead of a signed "fake proper motion" vector. The old ``as_ECL`` path could assign a negative ELONG/ELAT uncertainty (breaking model construction) after an ecliptic↔ICRS round trip, and both directions returned the wrong marginal σ after a non-trivial frame rotation. Correlations induced by conversion are not retained because timing-model parameters store only marginal uncertainties.
- Place ``solar_windx`` before the binary in ``DEFAULT_ORDER`` so SolarWindDispersionX delays and derivatives chain-rule through the binary the same way as ``solar_wind``.
- DDR batched-derivative cache keys now fingerprint upstream delay and parameter values in ``float64``, so 80-bit ``longdouble`` ulp jitter does not force a fresh Dual batch per column.
### Removed
