import os
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import poisson as scipy_poisson
from tqdm import trange

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

os.makedirs('figs/exploration/lognormal_poisson_check', exist_ok=True)

_proc = psutil.Process()


# def mem_gb():
    # """Return current process RSS in MB."""
    # return _proc.memory_info().rss / 1024**3


def mem_gb():
    """Return current process RSS (CPU RAM) in MB, and GPU memory if available."""
    cpu_mb = _proc.memory_info().rss / 1024**3
    print(f"CPU RAM used: {cpu_mb:.1f} GB")

    try:
        import pynvml
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # info = pynvml.nvmlMemoryInfo(handle)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mb = info.used / 1024**3
            print(f"GPU {i} memory used: {gpu_mb:.1f} GB")
    except ImportError:
        print("GPU memory: pynvml not installed (run `pip install pynvml`)")
    except pynvml.NVMLError as e:
        print(f"GPU memory: unavailable ({e})")

    return cpu_mb

def do_mem_check(message=None):
    if message is not None:
        print("="*100)
        print(message)
        print("="*100)
    mem_gb()
    
def df_memory_use(df):
    df_mem_use = df.memory_usage(deep=True).sum() / 1024**2  # in MB
    print("Memory usage for the dataframe: %2.2f MB"%(df_mem_use))
    

do_mem_check("Checkpoint 0.....")

# ── load & bin (identical setup to other scripts) ─────────────────────────────
# hod_df    = pd.read_csv('./data/hod_z0.csv')
hod_df    = pd.read_csv('./data/hod_z0_small.csv')
MASS_COL  = 'halo_mp'
log_edges = np.linspace(11., 15., 21)
edges     = 10**log_edges
bin_cents = 10**((log_edges[:-1] + log_edges[1:]) / 2)
n_bins    = len(bin_cents)   # 20


hod_df['mass_bin'] = pd.cut(hod_df[MASS_COL], bins=edges, labels=False)

df_sf_cen = hod_df[hod_df['N_cen_SF'] == 1].copy()
df_q_cen  = hod_df[hod_df['N_cen_Q']  == 1].copy()
subsets   = {'SF central': df_sf_cen, 'Q central': df_q_cen}
del hod_df

do_mem_check("Checkpoint 1.....")
# ── MCMC settings ─────────────────────────────────────────────────────────────
PRIOR_WIDTH = 0.5   # log-scale width for LogNormal priors
NUM_WARMUP  = 400
NUM_SAMPLES = 200
rng_key     = jax.random.PRNGKey(42)

do_mem_check("Checkpoint 2.....")
# ── method-of-moments helpers ─────────────────────────────────────────────────
def sigma_mom(mu, var):
    """
    MoM sigma from mean and variance of N.

    From  Var(N) = rate + rate^2*(exp(sigma^2)-1)  with  rate = mu:
      exp(sigma^2) = 1 + (Fano-1)/mu   =>   sigma = sqrt(log(...))
    """
    if mu <= 0.:
        return 0.1
    fano = var / mu
    arg  = max(1. + (fano - 1.) / mu, 1e-6)
    return float(np.sqrt(max(np.log(arg), 0.)))

def rho_c_mom(mu_sf, mu_q, cov_data, sig_sf, sig_q):
    """
    MoM rho_c from observed covariance.

    From  Cov(N_SF,N_Q) = rate_SF*rate_Q*(exp(rho_c*sigma_SF*sigma_Q)-1):
      rho_c = log(1 + Cov_data/(mu_sf*mu_q)) / (sig_sf*sig_q)
    """
    if sig_sf <= 0. or sig_q <= 0. or mu_sf <= 0. or mu_q <= 0.:
        return 0.
    arg = 1. + cov_data / (mu_sf * mu_q)
    arg = max(arg, 1e-6)
    rho = float(np.log(arg) / (sig_sf * sig_q))
    return float(np.clip(rho, -0.99, 0.99))

# ── NumPyro joint lognormal-Poisson model ─────────────────────────────────────
#
# Generative model per halo:
#   [x_SF, x_Q] ~ MVN(0, [[1, rho_c], [rho_c, 1]])
#   lambda_i  = exp(sigma_i * x_i  -  0.5*sigma_i^2)  [E[lambda_i]=1 by design]
#   N_sat_i  ~ Poisson(rate_i * lambda_i)
#
# Implied moments:
#   E[N_i]             = rate_i
#   Var(N_i)           = rate_i + rate_i^2 * (exp(sigma_i^2) - 1)
#   Fano(N_i)          = 1 + rate_i * (exp(sigma_i^2) - 1)
#   Cov(N_SF, N_Q)     = rate_SF * rate_Q * (exp(rho_c*sig_SF*sig_Q) - 1)
#
# Parameters: rate_SF, rate_Q > 0;  sigma_SF, sigma_Q > 0;  rho_c in (-1,1)
# Latent per halo: [x_SF, x_Q]  (n 2-vectors drawn jointly from MVN)

def make_joint_model(rate_sf_init, rate_q_init, sig_sf_init, sig_q_init, rho_c_init):
    def model(data_sf, data_q):
        n = len(data_sf)

        # ── global parameters ─────────────────────────────────────────────────
        rate_sf  = numpyro.sample('rate_sf',
                                  dist.LogNormal(jnp.log(rate_sf_init), PRIOR_WIDTH))
        rate_q   = numpyro.sample('rate_q',
                                  dist.LogNormal(jnp.log(rate_q_init),  PRIOR_WIDTH))
        sigma_sf = numpyro.sample('sigma_sf',
                                  dist.LogNormal(jnp.log(sig_sf_init),  PRIOR_WIDTH))
        sigma_q  = numpyro.sample('sigma_q',
                                  dist.LogNormal(jnp.log(sig_q_init),   PRIOR_WIDTH))

        # rho_c via tanh reparameterisation: rho_c_raw ∈ ℝ → rho_c ∈ (-1,1)
        rho_c_raw = numpyro.sample(
            'rho_c_raw',
            dist.Normal(jnp.arctanh(jnp.array(rho_c_init)), PRIOR_WIDTH))
        rho_c = numpyro.deterministic('rho_c', jnp.tanh(rho_c_raw))

        # ── per-halo latent variables drawn jointly from bivariate normal ─────
        cov = jnp.array([[1., rho_c], [rho_c, 1.]])
        with numpyro.plate('halos', n):
            x = numpyro.sample('x', dist.MultivariateNormal(jnp.zeros(2), cov))
            x_sf = x[..., 0]
            x_q  = x[..., 1]

            lam_sf = jnp.exp(sigma_sf * x_sf - 0.5 * sigma_sf**2)
            lam_q  = jnp.exp(sigma_q  * x_q  - 0.5 * sigma_q**2)

            numpyro.sample('N_sat_SF', dist.Poisson(rate_sf * lam_sf), obs=data_sf)
            numpyro.sample('N_sat_Q',  dist.Poisson(rate_q  * lam_q),  obs=data_q)

    return model

# ── NUTS fits: both central types × 40 mass bins ──────────────────────────────
fits = {}

for cen_label, df in subsets.items():
    print(f"\nFitting Joint LN-Poisson  [{cen_label}]")
    bin_fits = []

    for b in trange(n_bins):
        sub    = df[df['mass_bin'] == b]
        df_memory_use(sub)
        arr_sf = np.array(sub['N_sat_SF'].values, dtype=np.int32)
        arr_q  = np.array(sub['N_sat_Q'].values,  dtype=np.int32)
        n      = len(arr_sf)
        mass_mean = float(sub[MASS_COL].mean()) if n > 0 else np.nan

        mu_sf = float(arr_sf.mean())
        mu_q  = float(arr_q.mean())

        if mu_sf == 0. and mu_q == 0.:
            bin_fits.append(None)
            print(f"  bin {b:2d}: skipped  (n={n})")
            continue

        var_sf   = float(arr_sf.var())
        var_q    = float(arr_q.var())
        cov_data = float(np.cov(arr_sf.astype(float), arr_q.astype(float))[0, 1])
        
        # MoM initialisations (clamped to avoid degenerate starts)
        rate_sf_0 = max(mu_sf,  1e-3)
        rate_q_0  = max(mu_q,   1e-3)
        sig_sf_0  = max(sigma_mom(mu_sf, var_sf), 1e-3)
        sig_q_0   = max(sigma_mom(mu_q,  var_q),  1e-3)
        rho_c_0   = rho_c_mom(mu_sf, mu_q, cov_data, sig_sf_0, sig_q_0)

        print("="*100)
        print("rate_sf_0: %2.3f"%(rate_sf_0))
        print("rate_q_0: %2.3f"%(rate_q_0))
        print("sig_sf_0: %2.3f"%(sig_sf_0))
        print("sig_q_0: %2.3f"%(sig_q_0))
        print("cov_data: %2.3f"%(cov_data))
        print("rho_c_0: %2.3f"%(rho_c_0))
        print("="*100)
        rng_key, rng_ = jax.random.split(rng_key)
        model  = make_joint_model(rate_sf_0, rate_q_0, sig_sf_0, sig_q_0, rho_c_0)
        kernel = NUTS(model, init_strategy=init_to_value(values={
            'rate_sf'  : jnp.array(rate_sf_0),
            'rate_q'   : jnp.array(rate_q_0),
            'sigma_sf' : jnp.array(sig_sf_0),
            'sigma_q'  : jnp.array(sig_q_0),
            'rho_c_raw': jnp.array(float(np.arctanh(rho_c_0))),
        }))
        # Note: 'x' (per-halo MVN latents) is not initialised here;
        # NUTS draws its own starting point for the latent halo vectors.
        mcmc = MCMC(kernel, num_warmup=NUM_WARMUP,
                    num_samples=NUM_SAMPLES, progress_bar=True)
        mcmc.run(rng_, jnp.array(arr_sf), jnp.array(arr_q))
        s = mcmc.get_samples()
        rate_sf_s    = float(np.array(s['rate_sf']).mean())
        rate_sf_std  = float(np.array(s['rate_sf']).std())
        rate_q_s     = float(np.array(s['rate_q']).mean())
        rate_q_std   = float(np.array(s['rate_q']).std())
        sigma_sf_s   = float(np.array(s['sigma_sf']).mean())
        sigma_sf_std = float(np.array(s['sigma_sf']).std())
        sigma_q_s    = float(np.array(s['sigma_q']).mean())
        sigma_q_std  = float(np.array(s['sigma_q']).std())
        rho_c_s      = float(np.array(s['rho_c']).mean())
        rho_c_std    = float(np.array(s['rho_c']).std())
        # print(f"  bin {b:2d}: memory before del mcmc/samples: {mem_gb():.1f} GB")
        del mcmc, s
        # print(f"  bin {b:2d}: memory after  del mcmc/samples: {mem_gb():.1f} GB")

        print(f"  bin {b:2d}: n={n:7d}  "
              f"rate_sf={rate_sf_s:.3f}  rate_q={rate_q_s:.3f}  "
              f"sig_sf={sigma_sf_s:.3f}  sig_q={sigma_q_s:.3f}  "
              f"rho_c={rho_c_s:.3f}")

        bin_fits.append({
            'n'         : n,
            'mass_mean' : mass_mean,
            'data_sf'   : arr_sf,
            'data_q'    : arr_q,
            'rate_sf'   : rate_sf_s,   'rate_sf_std'  : rate_sf_std,
            'rate_q'    : rate_q_s,    'rate_q_std'   : rate_q_std,
            'sigma_sf'  : sigma_sf_s,  'sigma_sf_std' : sigma_sf_std,
            'sigma_q'   : sigma_q_s,   'sigma_q_std'  : sigma_q_std,
            'rho_c'     : rho_c_s,     'rho_c_std'    : rho_c_std,
        })

    fits[cen_label] = bin_fits

    # ── save per-bin parameter summary to CSV ─────────────────────────────────
    _params = ['rate_sf', 'rate_q', 'sigma_sf', 'sigma_q', 'rho_c']
    rows = []
    for b, f in enumerate(bin_fits):
        if f is None:
            row = {'bin': b, 'log_mass_mean': np.nan, 'n': 0}
            for p in _params:
                row[p] = np.nan
                row[f'{p}_std'] = np.nan
        else:
            row = {
                'bin'          : b,
                'log_mass_mean': np.log10(f['mass_mean']),
                'n'            : f['n'],
            }
            for p in _params:
                row[p] = f[p]
                row[f'{p}_std'] = f[f'{p}_std']
        rows.append(row)

    cen_tag = 'sf_cen' if 'SF' in cen_label else 'q_cen'
    csv_path = f'figs/exploration/lognormal_poisson_check/params_{cen_tag}.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved parameter summary → {csv_path}")


# ── marginal PMF via Monte Carlo integration ──────────────────────────────────
_N_MC    = 80_000
_MC_SEED = 0


def poisson_lognormal_pmf(k_arr, rate, sigma):
    """
    P(N=k) for Poisson(rate*lambda), lambda ~ LogNormal(mean=1),
    estimated by averaging Poisson PMFs over MC draws of lambda.
    """
    rng    = np.random.default_rng(_MC_SEED)
    z      = rng.standard_normal(_N_MC)
    lam    = np.exp(sigma * z - 0.5 * sigma**2)
    mu_arr = rate * lam                    # shape (_N_MC,)
    # vectorised: (len(k_arr), _N_MC) → mean over MC axis
    pmf = scipy_poisson.pmf(k_arr[:, None], mu_arr[None, :]).mean(axis=1)
    return pmf


# ── PMF comparison grid (marginals) ──────────────────────────────────────────
def plot_pmf_grid(cen_label, sat_key, bin_fits, bin_cents, color, fname):
    """
    4×5 panel grid — one per mass bin.
    Shows empirical marginal PMF vs fitted Poisson-lognormal marginal.

    sat_key : 'data_sf' or 'data_q'
    """
    rate_key  = 'rate_sf'  if sat_key == 'data_sf' else 'rate_q'
    sigma_key = 'sigma_sf' if sat_key == 'data_sf' else 'sigma_q'
    sat_sym   = (r'N_\mathrm{sat}^\mathrm{SF}' if sat_key == 'data_sf'
                 else r'N_\mathrm{sat}^\mathrm{Q}')

    fig, axes = plt.subplots(4, 5, figsize=(16, 14))

    for idx, ax in enumerate(axes.flat):
        fit  = bin_fits[idx]
        logM = np.log10(bin_cents[idx])
        ax.set_title(rf'$\log M={logM:.2f}$', fontsize=7)
        ax.tick_params(labelsize=6)

        if fit is None:
            ax.text(0.5, 0.5, 'too few\nhalos',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='grey')
            continue

        rate  = fit[rate_key]
        sigma = fit[sigma_key]
        data  = fit[sat_key]
        n     = fit['n']

        # x range from fitted model (mean ± 7σ of marginal N)
        var_n = rate + rate**2 * (np.exp(sigma**2) - 1.)
        N_max = int(np.ceil(rate + 7. * np.sqrt(max(var_n, 0.))))
        k_arr = np.arange(0, N_max + 1)

        # empirical PMF (skip zero-count bins for log scale)
        counts  = np.bincount(data, minlength=N_max + 1)[:N_max + 1]
        emp_pmf = counts / counts.sum()
        mask    = emp_pmf > 0

        model_pmf = poisson_lognormal_pmf(k_arr, rate, sigma)

        ax.bar(k_arr[mask], emp_pmf[mask], color=color, alpha=0.65,
               width=0.5, label='Data', zorder=2)
        ax.plot(k_arr, model_pmf, color='k', ls='-', lw=1.2,
                label='LN-P fit', zorder=3)

        ax.set_yscale('log')
        ax.set_xlim(-0.5, N_max + 0.5)
        ax.set_ylim(bottom=0.3 / n)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.annotate(f'n={n}', xy=(0.97, 0.97), xycoords='axes fraction',
                    ha='right', va='top', fontsize=5.5)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    fig.suptitle(rf'PMF of ${sat_sym}$ — {cen_label} — Data vs LN-Poisson fit',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f'figs/exploration/lognormal_poisson_check/{fname}', dpi=150)
    plt.close()


for cen_label in ['SF central', 'Q central']:
    cen_tag = 'sf_cen' if 'SF' in cen_label else 'q_cen'
    bf      = fits[cen_label]
    for sat_key, sat_tag, color in [
        ('data_sf', 'sf_sat', 'steelblue'),
        ('data_q',  'q_sat',  'firebrick'),
    ]:
        print(f"\nPlotting PMF grid: {cen_label} | {sat_key}")
        plot_pmf_grid(cen_label, sat_key, bf, bin_cents, color,
                      fname=f'pmf_{sat_tag}_{cen_tag}.png')

# free raw data arrays — no longer needed after PMF plots
for bf in fits.values():
    for f in bf:
        if f is not None:
            del f['data_sf'], f['data_q']

# ── model-predicted statistics from fit parameters ────────────────────────────
def fit_stats(fit):
    """Return (mean_sf, mean_q, fano_sf, fano_q, corr) from posterior means."""
    r_sf  = fit['rate_sf'];  r_q  = fit['rate_q']
    s_sf  = fit['sigma_sf']; s_q  = fit['sigma_q']
    rho_c = fit['rho_c']

    var_sf  = r_sf + r_sf**2 * (np.exp(s_sf**2) - 1.)
    var_q   = r_q  + r_q**2  * (np.exp(s_q**2)  - 1.)
    fano_sf = var_sf / r_sf if r_sf > 0 else np.nan
    fano_q  = var_q  / r_q  if r_q  > 0 else np.nan
    cov_mod = r_sf * r_q * (np.exp(rho_c * s_sf * s_q) - 1.)
    corr    = (cov_mod / np.sqrt(var_sf * var_q)
               if var_sf > 0 and var_q > 0 else np.nan)

    return r_sf, r_q, fano_sf, fano_q, corr


# ── summary plot: mean & Fano ─────────────────────────────────────────────────
def plot_summary(fits, subsets, bin_cents):
    """
    2 rows × 2 cols (height_ratios [3,1]).
      Col 0 : N_sat_SF       Col 1 : N_sat_Q
      Row 0 : mean           Row 1 : Fano factor
    Data: solid line; LN-P fit: dashed line.
    """
    eps    = 0.04
    x_data = bin_cents * (1. - eps)
    x_fit  = bin_cents * (1. + eps)

    cen_props = {
        'SF central': dict(color='steelblue', mk_data='o', mk_fit='^'),
        'Q central' : dict(color='firebrick',  mk_data='s', mk_fit='D'),
    }
    sat_specs = [
        ('N_sat_SF', 'rate_sf', 'sigma_sf', r'$N_\mathrm{sat}^\mathrm{SF}$'),
        ('N_sat_Q',  'rate_q',  'sigma_q',  r'$N_\mathrm{sat}^\mathrm{Q}$'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [3, 1]},
                             sharex='col')

    for col_idx, (sat_col, rate_key, sigma_key, sat_label) in enumerate(sat_specs):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        for cen_label, props in cen_props.items():
            color = props['color']
            df    = subsets[cen_label]
            bf    = fits[cen_label]

            # empirical stats
            grp       = df.groupby('mass_bin')[sat_col]
            mean_data = grp.mean().reindex(range(n_bins)).values
            var_data  = grp.var(ddof=1).reindex(range(n_bins)).values
            fano_data = var_data / mean_data

            # model-predicted stats
            rate_fit = np.array([
                f[rate_key] if f is not None else np.nan for f in bf])
            fano_fit = np.array([
                1. + f[rate_key] * (np.exp(f[sigma_key]**2) - 1.)
                if f is not None else np.nan for f in bf])

            ax_top.loglog(x_data, mean_data, color=color, ls='-',
                          marker=props['mk_data'], ms=4,
                          label=f'{cen_label} data')
            ax_top.loglog(x_fit, rate_fit, color=color, ls='--',
                          marker=props['mk_fit'], ms=4,
                          label=f'{cen_label} LN-P fit')

            ax_bot.semilogx(x_data, fano_data, color=color, ls='-',
                            marker=props['mk_data'], ms=4)
            ax_bot.semilogx(x_fit, fano_fit, color=color, ls='--',
                            marker=props['mk_fit'], ms=4)

        ax_top.set_ylabel('Mean occupation')
        ax_top.set_title(sat_label)
        ax_top.legend(fontsize=8)
        ax_top.grid(True, which='both', ls=':', alpha=0.4)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        ax_bot.axhline(1, color='k', ls=':', lw=0.8, label='Poisson (=1)')
        ax_bot.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
        ax_bot.set_ylabel('Variance / Mean')
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, which='both', ls=':', alpha=0.4)

    fig.suptitle('HOD satellites — LN-Poisson fit: Mean & Fano', fontsize=13)
    plt.tight_layout()
    plt.savefig('figs/exploration/lognormal_poisson_check/summary.png', dpi=150)
    plt.close()


# ── cross-correlation plot: data vs model ─────────────────────────────────────
def plot_cross_correlation(fits, subsets, bin_cents):
    """
    Pearson r between N_sat_SF and N_sat_Q: data (solid) vs LN-P fit (dashed).
    One pair of curves per central type.
    """
    eps    = 0.04
    x_data = bin_cents * (1. - eps)
    x_fit  = bin_cents * (1. + eps)

    cen_props = {
        'SF central': dict(color='steelblue', mk_data='o', mk_fit='^', ls='-'),
        'Q central' : dict(color='firebrick',  mk_data='s', mk_fit='D', ls='--'),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color='k', ls=':', lw=0.8)

    for cen_label, props in cen_props.items():
        color = props['color']
        df    = subsets[cen_label]
        bf    = fits[cen_label]

        # empirical Pearson r per bin
        corr_data = (df.groupby('mass_bin')
                       .apply(lambda g: g['N_sat_SF'].corr(g['N_sat_Q']))
                       .reindex(range(n_bins))
                       .values)

        # model-predicted Pearson r from fit parameters
        corr_fit = np.array([
            fit_stats(f)[4] if f is not None else np.nan for f in bf])

        ax.semilogx(x_data, corr_data, color=color, ls=props['ls'],
                    marker=props['mk_data'], ms=4, label=f'{cen_label} data')
        ax.semilogx(x_fit, corr_fit, color=color, ls=':',
                    marker=props['mk_fit'], ms=4, label=f'{cen_label} LN-P fit')

    ax.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
    ax.set_ylabel(r'Pearson $r$')
    ax.set_title(r'Cross-correlation: $N_\mathrm{sat}^\mathrm{SF}$ vs '
                 r'$N_\mathrm{sat}^\mathrm{Q}$ — Data vs LN-Poisson fit')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig('figs/exploration/lognormal_poisson_check/cross_correlation.png',
                dpi=150)
    plt.close()


# ── run plots ─────────────────────────────────────────────────────────────────
print("\nPlotting summary...")
plot_summary(fits, subsets, bin_cents)

print("Plotting cross-correlation...")
plot_cross_correlation(fits, subsets, bin_cents)
del subsets

print("\nDone. Figures saved to figs/exploration/lognormal_poisson_check/")
