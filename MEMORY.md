# Joint Distribution Project Memory

## Project Overview
Astrophysics pipeline fitting joint galaxy count distributions across density bins.
Working dir: `/home2/supranta/MBI/joint_distribution/`
Conda env: `jupyter_gpu`
Data: `/home2/supranta/MBI/data/slab_data/z0_slab_data.h5`

## Current Architecture (post-refactor)

### Module: `galaxy_models/` (renamed from `gp_models/`)
- **models.py** — N-type NumPyro models; all accept `counts: (N_types, N_pix)` JAX array.
  Uses `dist.Independent(dist.Poisson(mu.T), 1)` with `obs=counts.T` pattern to avoid numpyro plate warnings.
- **fitting.py** — Generic `fit_nuts(model_name, counts, ...)` → standardized result dict.
  `_infer_flags(model_name)` derives `(is_lognormal, shared_latent, exp_link)` from key string.
- **stats.py** — Per-type PMF stats (unchanged signatures), generalized summary stats (S, N_types), posterior_predictive returns `(counts_pp, corr_matrix)`.
- **config.py** — `load_config(path)` returns plain dict from YAML.
- **__init__.py** — exports fit_nuts, MODELS, load_config, stats functions, plotting.

### Entry Point: `run_fits.py`
Usage: `python run_fits.py configs/example.yaml`
Config specifies: data.path, data.h5_keys (list of [group, dataset] pairs), model, fitting params, output.out_dir

### Config Format (`configs/`)
```yaml
data:
    path: ../data/slab_data/z0_slab_data.h5
    h5_keys:
        - ["UM star-forming", "Ng"]
        - ["UM quenched", "Ng"]
model: joint_lognormal
fitting:
    n_bins: 40
    num_warmup: 500
    num_samples: 500
    mc_n_lam: 2000
output:
    out_dir: figs/joint_lognormal_fits
```

### Output Structure
- `<out_dir>/fitted_params.npz` — disp_samples (N_bins, S, N_types), rate_samples, scalar stats
- `<out_dir>/lam/type{t}_bin_{i:02d}.npy`
- `<out_dir>/bins/bin_{i:02d}.npz`

### Model Registry (`MODELS` dict keys)
gp_joint, egp_joint, gp_independent, egp_independent, gp_joint_scaling, egp_joint_scaling, joint_lognormal, joint_lognormal_exp

### H5 File Structure
Keys: 'UM quenched', 'UM star-forming', 'MagLim centrals', 'MagLim satellites', 'RedMagic centrals', 'RedMagic satellites', 'delta_2d'
Each galaxy key has subkey 'Ng' with 2D array.

### plot_fits.py
Updated to use galaxy_models. Now takes `--model` (label) and `--out-dir` args instead of MODEL_REGISTRY. Marked TODO for N-type support.

### Removed files
- `fit_models.py`, `fit_joint_gp.py` (superseded by run_fits.py)
- `gp_models/` is kept for backward compat but new code uses `galaxy_models/`

## Key Design Patterns
- `_infer_flags("joint_lognormal")` → `(True, True, False)` (lognormal, shared, no exp-link)
- `_infer_flags("egp_joint")` → `(False, True, True)`
- Models: `gp_*` = GP family, `egp_*` = EGP family, `joint_lognormal*` = log-normal family
- Result dict: disp_samples (S, N_types), rate_samples (S, N_types), lam_mean (N_types, N_pix)
