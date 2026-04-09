# Changelog

All notable changes to Nautilus are documented here.

## [0.5.8] — 2026-04-09

### Fixed
- **Data discovery**: `config.py` now finds bundled CSVs inside
  `site-packages/nautilus/data/` for non-editable PyPI installs, ending
  the `AppData` fallback that caused `FileNotFoundError` on first run.
- **Cache directory**: Cache is written to `~/.cache/nautilus/` when running
  from a pip-installed package, preventing writes to read-only site-packages.
- **Notebook double-shift** (cells 10, 22, 25): `compute_price_above_ma`
  already applies `shift(1)` internally — extra `.shift(1)` calls removed.
- **Notebook cell 24**: `fit_hmm` was receiving the pre-shifted `macro_df`
  instead of the raw frame, causing a 2-day lag on all macro HMM features.
  Now correctly passes `macro_raw`.
- **Notebook cell 28**: `(1 + log_ret).cumprod()` replaced with
  `price / price.iloc[0]` — log returns require `exp(cumsum)`, not
  `cumprod(1 + r)`.

### Changed
- **PyPI packaging**: `src/nautilus/data/*.csv` bundled as package data via
  `[tool.setuptools.package-data]`; `MANIFEST.in` added for sdist.
- **CLI entry point**: `nautilus` command added — runs `streamlit run` on the
  dashboard without needing to know the install path.
- **Dashboard**: decorative emoji removed from sidebar section headers,
  buttons, section dividers, and expander labels. Regime colour/emoji markers
  in data tables retained (they carry semantic meaning).
- **Version bump**: `0.5.0` → `0.5.5`.

## [0.5.0] — 2026-04-09

- Initial public release candidate.
