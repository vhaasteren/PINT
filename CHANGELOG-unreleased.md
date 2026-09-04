# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- TCB/TDB conversion now matches PINT's IAU 2006/Astropy TDB forward model,
  keeps radio frequency undilated, converts FD/FDJUMP and order-aware DM
  coefficients, leaves PX and UTC/data-span selectors explicitly invariant,
  and reports unsupported active deterministic terms. Converted epochs and DM
  values therefore differ from the previous legacy-IFTE conversion. The
  compatibility alias `IFTE_K` now denotes the IAU/ERFA rate; the obsolete
  `IFTE_MJD0` and `IFTE_KM1` module constants have been removed.
### Added
- Time-domain solar wind GP noise components: ridge, squared-exponential, Matérn, and quasi-periodic kernels
- Documentation page explaining the time-domain solar wind noise model, its interpolation basis, and how it differs from the Fourier-basis noise models
- `TOAs.get_tdb_seconds()`, returning the TDB times of the TOAs in seconds with a selectable dtype
### Fixed
### Removed
