# Nautilus BTC — Windows Setup Guide

## Prerequisites

- **Python 3.11+** — download from https://python.org (tick "Add to PATH")
- **Git** (optional) — https://git-scm.com

The C++ layer requires Visual Studio Build Tools to compile, but is **not required** — the system works fully in Python-only mode on Windows.

---

## Quickstart (PowerShell)

```powershell
# 1. Open PowerShell in the nautilus_btc\ folder
#    (Right-click inside folder → "Open in Terminal")

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Create required directories
mkdir logs, models -Force

# 5. Validate the stack (no internet needed)
python scripts\smoke_test.py

# 6. Run tests
python -m pytest tests\ -v

# 7. Launch live dashboard
python main.py

# 8. Backtest only (fetches Binance history, fits HMM, prints table)
python main.py --backtest

# 9. Walk-forward optimisation + dashboard
python main.py --optimise
```

## Or use setup.bat (CMD / PowerShell)

```powershell
.\setup.bat
```

---

## Notes

- **No `chmod` or `source` on Windows** — these are Linux/macOS commands.
  Use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

- **C++ layer (optional)**: Requires Visual Studio 2022 Build Tools + CMake.
  Without it, the system automatically uses the Python bar builder.
  All features work identically; throughput is lower (~30K vs ~300K msgs/s).

  To build on Windows (MSYS2 MinGW recommended):
  ```bash
  # In MSYS2 UCRT64 terminal:
  pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-gcc
  bash build_cpp.sh
  ```

- **Textual dashboard** renders correctly in Windows Terminal (Win11)
  and VS Code terminal. Avoid legacy `cmd.exe` which has limited Unicode.

- **asyncio on Windows**: Python 3.12 uses `ProactorEventLoop` by default.
  This is compatible. If you see `NotImplementedError` from asyncio,
  add this at the top of `main.py`:
  ```python
  import asyncio, sys
  if sys.platform == "win32":
      asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  ```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'aiohttp'` | `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'textual'` | `pip install textual rich` |
| Dashboard renders garbled | Use Windows Terminal, not cmd.exe |
| `pytest-asyncio` warning about `asyncio_mode` | Harmless on Windows; tests still pass |
