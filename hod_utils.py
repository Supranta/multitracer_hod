"""
Shared utilities for the joint HOD pipeline.

Provides data loading, mass binning, galaxy subset splitting, and memory
diagnostics.  All functions are pure or clearly documented side-effectful.
"""

import psutil
import numpy as np
import pandas as pd

from hod_config import MASS_COL, EDGES, N_BINS, DATA_FILE_FULL, DATA_FILE_SMALL

_proc = psutil.Process()

# ── Memory diagnostics ────────────────────────────────────────────────────────

def mem_gb() -> float:
    """Return current process RSS in GB, and print GPU memory if available."""
    cpu_gb = _proc.memory_info().rss / 1024**3
    print(f"CPU RAM used: {cpu_gb:.1f} GB")

    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {i} memory used: {info.used / 1024**3:.1f} GB")
    except ImportError:
        print("GPU memory: pynvml not installed (run `pip install pynvml`)")
    except pynvml.NVMLError as e:
        print(f"GPU memory: unavailable ({e})")

    return cpu_gb


def do_mem_check(message: str | None = None) -> None:
    """Print a labelled checkpoint with current CPU and GPU memory usage."""
    if message is not None:
        print("=" * 100)
        print(message)
        print("=" * 100)
    mem_gb()


def df_memory_use(df: pd.DataFrame) -> None:
    """Print the in-memory size of a DataFrame in MB."""
    mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"DataFrame memory usage: {mb:.2f} MB")


# ── Data loading ──────────────────────────────────────────────────────────────

_REQUIRED_COLUMNS = {'halo_mp', 'N_cen_SF', 'N_cen_Q', 'N_sat_SF', 'N_sat_Q'}


def assign_mass_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with a 'mass_bin' column added.

    Bins are defined by EDGES in hod_config (40 log-spaced bins, 10^11–10^15).
    Rows outside the bin range get NaN and are left in the DataFrame.
    """
    assert MASS_COL in df.columns, f"Expected column '{MASS_COL}' not found in DataFrame"
    result = df.copy()
    result['mass_bin'] = pd.cut(result[MASS_COL], bins=EDGES, labels=False)
    return result


def load_hod_dataframe(use_full: bool = False) -> pd.DataFrame:
    """
    Load the HOD summary table and assign mass bins.

    Parameters
    ----------
    use_full : bool
        If True, load the full catalogue (hod_z0.csv).
        If False (default), load the 10% subsample (hod_z0_small.csv).

    Returns
    -------
    pd.DataFrame with columns including halo_mp, N_cen_SF, N_cen_Q,
    N_sat_SF, N_sat_Q, mass_bin.
    """
    path = DATA_FILE_FULL if use_full else DATA_FILE_SMALL
    df   = pd.read_csv(path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing expected columns in {path}: {missing}"
    assert len(df) > 0, f"Loaded an empty DataFrame from {path}"

    return assign_mass_bins(df)


# ── Galaxy subset splitting ───────────────────────────────────────────────────

def split_by_central_type(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split the HOD DataFrame into SF-central and Q-central subsets.

    Returns
    -------
    dict with keys 'SF central' and 'Q central', each a filtered copy.
    """
    subsets = {
        'SF central': df[df['N_cen_SF'] == 1].copy(),
        'Q central' : df[df['N_cen_Q']  == 1].copy(),
    }
    for label, sub in subsets.items():
        assert len(sub) > 0, f"Subset '{label}' is empty after filtering"
    return subsets
