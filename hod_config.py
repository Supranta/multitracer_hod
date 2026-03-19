"""
Shared constants for the joint HOD pipeline.

All mass-binning parameters, data file paths, and plotting styles are defined
here once and imported everywhere else.  Nothing in this module has side
effects: it only defines values.
"""

import numpy as np

# ── Mass column ───────────────────────────────────────────────────────────────
# Peak halo mass is the preferred mass variable for HOD work (more stable than
# current mass across cosmic time).
MASS_COL = 'halo_mp'

# ── Mass-bin grid ─────────────────────────────────────────────────────────────
# 40 equal-width bins in log10(M) from 10^11 to 10^15 M_sun/h.
# N_MASS_BINS = 40 is the single authoritative value; all scripts must import
# it rather than hard-coding their own number of bins.
LOG_MASS_MIN  = 11.0
LOG_MASS_MAX  = 15.0
N_MASS_BINS   = 40

LOG_EDGES = np.linspace(LOG_MASS_MIN, LOG_MASS_MAX, N_MASS_BINS + 1)
EDGES     = 10.0 ** LOG_EDGES
BIN_CENTS = 10.0 ** ((LOG_EDGES[:-1] + LOG_EDGES[1:]) / 2.0)
N_BINS    = len(BIN_CENTS)   # == N_MASS_BINS

# ── Data file paths ───────────────────────────────────────────────────────────
# Use DATA_FILE_SMALL for exploration/quick runs; DATA_FILE_FULL for production.
# Scripts select between them via load_hod_dataframe(use_full=...) in hod_utils.
DATA_FILE_FULL  = './data/hod_z0.csv'
DATA_FILE_SMALL = './data/hod_z0_small.csv'

# ── Plotting styles ───────────────────────────────────────────────────────────
# One definition shared across all scripts.  Keys are the central-type labels
# used throughout the pipeline ('SF central', 'Q central').

# Per-type color, data marker, and fit marker.
CEN_PROPS = {
    'SF central': dict(color='steelblue', mk_data='o', mk_fit='^', ls='-'),
    'Q central' : dict(color='firebrick', mk_data='s', mk_fit='D', ls='--'),
}

# Line style and marker for data vs model curves (used in data_exploration.py).
CENTRAL_STYLE = {
    'SF central': ('-',  'o'),
    'Q central' : ('--', 's'),
}

COLOR_MAP = {
    'SF central': 'steelblue',
    'Q central' : 'firebrick',
}
