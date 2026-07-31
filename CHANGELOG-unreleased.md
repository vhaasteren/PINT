# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Moved altitude calculation to TOAs object, to make it only happen once
- `WidebandDownhillFitter` now handles correlated noise correctly.
- `pintk` Diff/Unc calculation now uses post-fit uncertainties.
- Updated GMRT coordinates.
- Replaced custom ``pint.ls`` with astropy ``u.lsec``
- Updated code to remove deprecation warnings during CI
- Every active FBX model now uses FB coefficients as its complete orbital-phase representation. An ordinary `PBDOT` supplied with `FB0` is converted to `FB1` instead of being silently ignored. `PB` and applicable `PBDOT` values are exposed as derived views, so `as_parfile()` gains commented `# PB`/`# PBDOT` lines for existing pure-FBX models and no longer exposes an unset writable `PBDOT` that the FBX orbit would ignore.
- DDGR+FBX is now rejected explicitly before normalization. The new general PB/FB bridge does not broaden DDGR into a partially supported configuration or rely on incidental missing-FB0/PB-conflict failures. Correct support requires dynamic synchronization of DDGR's PB-dependent post-Keplerian quantities and their analytic `PB(FB0)` chain-rule derivatives.
- `BinaryDDH` now accepts a negative fitted `H3` with a warning when `STIGMA` is finite and positive. The DDH delay is smooth in the signed amplitude `H3`, but the resulting negative derived `M2` is not physically interpretable. Validation of free `M2`, generic `SINI`, and non-DDH binary models is unchanged.
### Added
- Plot whitened DM residuals in pintk.
- `ssb_to_psb_xyz_ECL` and `ssb_to_psb_xyz_ICRS` are now cached
- Binary orbital-phase normalization now accepts `PB` (and optionally ordinary `PBDOT`) together with `FBn` while `FB0` is omitted. PINT converts the nominal model to a canonical FBX Taylor series, inserts frozen zero coefficients for sparse indices, and exposes `PB`/`PBDOT` as read-only derived views. Every valued FB coefficient is now guaranteed to contribute to orbital phase or produce a clear error.
### Fixed
- Sparse FBX series no longer silently ignore coefficients after the first missing index.
- Prefer DD over BT when guessing the binary model for Tempo2 `T2` par files (`allow_T2`), matching Tempo2's `allTerms=1` behavior
- `WidebandTOAFitter` raises a warning if the model has correlated errors (It used to give wrong results before).
- Fixed bug where "include_bipm" flag was being ignored when loading Fermi TOAs with weights, now defaults to using EPHEM, CLOCK and PLANET_SHAPIRO from the timing model
- When flags are created based off jumps uses strings instead of None
- When writing tempo format parfiles, use 0 instead of inf for TZRFRQ
- Write VLBI frame rotation parameters correctly to par file. 
- Make `get_prefix_timeranges` work for SWX.
- Some of the `gridutils` functions had improper logging behavior
- Fixed bug in changing epoch for ELL1k model
- Fixed `gridutils` behavior for 1 CPU
- Fixed bug in `GaussianRV_gen`, where the probability distribution function was not normalized correctly. Changed to use `scipy.stats.truncnorm` instead of the custom `GaussianRV_gen`.
- Fixed `convert_binary()` for ELL1H models to run `setup()` and not use H4 when not desired
- Fixed bug in `model.compare()` where it failed for `PosixPath` objects
- Fixed bug in printing of parameter correlation/covariance matrices
- `make_fake_toas_fromMJDs` now does not assume `PLANET_SHAPIRO` is in the model - it checks.
- Make VLBI frame rotation work correctly when proper motion is present.
- Changed some API to pass Mac CI
### Removed
