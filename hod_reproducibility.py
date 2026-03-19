"""
Reproducibility helpers for the joint HOD pipeline.

Every MCMC script should call record_library_versions() and
save_mcmc_run_config() before its fitting loop so that every output CSV
is accompanied by a JSON sidecar documenting exactly what produced it.
"""

import sys
import json
import datetime
import importlib.metadata


# Libraries used anywhere in the pipeline.
_TRACKED_LIBRARIES = [
    'numpy', 'pandas', 'scipy', 'matplotlib',
    'jax', 'numpyro', 'tqdm', 'psutil', 'pynvml',
]


def _library_versions() -> dict[str, str]:
    """Return a dict mapping each tracked library name to its version string."""
    versions = {}
    for lib in _TRACKED_LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = 'not installed'
    return versions


def record_library_versions(output_path: str) -> None:
    """
    Write library versions and Python version to a JSON file.

    Includes a UTC timestamp so the file is self-dated.  Call this once at
    the top of each MCMC script, before any fitting begins.
    """
    record = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat(),
        'python'        : sys.version,
        'libraries'     : _library_versions(),
    }
    with open(output_path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"Library versions recorded → {output_path}")


def save_mcmc_run_config(output_path: str, config: dict) -> None:
    """
    Serialise MCMC hyperparameters and run metadata to a JSON sidecar file.

    Callers should include at minimum: num_warmup, num_samples, rng_seed,
    data_file, script_name.  A UTC timestamp is added automatically.

    Why: every output CSV should be reproducible from its sidecar alone —
    knowing the data file, the random seed, and the sampler settings is
    sufficient to re-run and obtain statistically equivalent results.
    """
    record = {
        'timestamp_utc': datetime.datetime.utcnow().isoformat(),
        **config,
    }
    with open(output_path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"MCMC run config saved → {output_path}")
