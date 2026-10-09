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
### Added
- `BINARY DDR` (Damour-Deruelle-Regular binary): `BinaryDDR` wrap around the stand-alone delay kernel; same `PulsarBinary` `FB0`-`FBn` tokens as DD/ELL1, with the kernel evaluating the full prefix (scaled Horner, centering, `DDR_FB_KMAX`; no mixed `PBDOT` / `DDRKINE Y`); analytic astrometric and batched full-model derivatives including the `-B_t A_θ` time-argument correction, strict unsupported-parameter validation, WLS recovery, and exact secular/gauge-aware `convert_binary` import/export with fit-space and uncertainty reporting. ELL1 `TASC` is translated by \(3/2\,xh\) with the phase chart and `A1` copied; ELL1H absorbed conversions apply the Freire-Wex orbit encode/decode once as the complete map.
- Binary orbital-phase normalization now accepts `PB` (and optionally ordinary `PBDOT`) together with `FBn` while `FB0` is omitted. PINT converts the nominal model to a canonical FBX Taylor series, inserts frozen zero coefficients for sparse indices, and exposes `PB`/`PBDOT` as read-only derived views. Every valued FB coefficient is now guaranteed to contribute to orbital phase or produce a clear error.
### Fixed
- Sparse FBX series no longer silently ignore coefficients after the first missing index.
- Propagate one-way astrometric marginal uncertainties in ``as_ECL`` / ``as_ICRS`` by diagonal covariance rotation instead of a signed "fake proper motion" vector. The old ``as_ECL`` path could assign a negative ELONG/ELAT uncertainty (breaking model construction) after an ecliptic↔ICRS round trip, and both directions returned the wrong marginal σ after a non-trivial frame rotation. Correlations induced by conversion are not retained because timing-model parameters store only marginal uncertainties.
- DDR batched-derivative cache keys now fingerprint upstream delay and parameter values in ``float64``, so 80-bit ``longdouble`` ulp jitter does not force a fresh Dual batch per column.
### Removed
