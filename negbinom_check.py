import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import nbinom as scipy_nbinom
from tqdm import trange

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from hod_config         import BIN_CENTS, N_BINS, DATA_FILE_FULL
from hod_utils          import load_hod_dataframe, split_by_central_type
from hod_reproducibility import record_library_versions, save_mcmc_run_config


os.makedirs('figs/exploration/negbinom_check', exist_ok=True)

# ── MCMC settings ─────────────────────────────────────────────────────────────
PRIOR_VAL   = 0.2
NUM_WARMUP  = 500
NUM_SAMPLES = 1000
RNG_SEED    = 42

# ── NegBin model ──────────────────────────────────────────────────────────────
# Factory: priors are LogNormal centred on per-bin empirical estimates.
# mu_init = empirical mean; r_init = method-of-moments estimate of concentration.

def make_nb_model(mu_init, r_init):
    def nb_model(data):
        mu = numpyro.sample('mu', dist.LogNormal(jnp.log(mu_init), PRIOR_VAL))
        r  = numpyro.sample('r',  dist.LogNormal(jnp.log(r_init),  PRIOR_VAL))
        numpyro.sample('obs', dist.NegativeBinomial2(mu, r), obs=data)
    return nb_model


def fit_negbinom_per_bin(subsets):
    """
    Run NUTS for NegBin(mu, r) independently per mass bin and central type.

    Returns a dict keyed by (cen_label, sat_col); each value is a list of
    length N_BINS with None (skipped) or dict(n, data, mu, r).
    """
    rng_key = jax.random.PRNGKey(RNG_SEED)
    fits    = {}

    for cen_label, df in subsets.items():
        for sat_col in ['N_sat_SF', 'N_sat_Q']:
            print(f"\nFitting NB  [{cen_label}  |  {sat_col}]")
            bin_fits = []

            for b in trange(N_BINS):
                sub  = df[df['mass_bin'] == b]
                data = jnp.array(sub[sat_col].values, dtype=jnp.int32)
                n    = len(data)
                assert data.shape[0] == n, f"Shape mismatch in bin {b}"

                if float(data.mean()) == 0.:
                    bin_fits.append(None)
                    print(f"  bin {b:2d}: skipped  (n={n})")
                    continue

                mu_emp  = float(data.mean())
                var_emp = float(data.var())
                denom   = var_emp - mu_emp
                # method-of-moments: r = mu^2 / (var - mu)
                # if var <= mu (sub/Poisson) the distribution is near-Poisson,
                # so cap r at a large value to avoid r -> inf
                r_mom   = float(np.clip(mu_emp**2 / denom, 1e-2, 1e3)
                                if denom > 0 else 1e3)

                rng_key, rng_key_current = jax.random.split(rng_key)
                model  = make_nb_model(jnp.array(mu_emp), jnp.array(r_mom))
                kernel = NUTS(model, init_strategy=init_to_value(
                                  values={'mu': jnp.array(mu_emp),
                                          'r' : jnp.array(r_mom)}))
                mcmc = MCMC(kernel, num_warmup=NUM_WARMUP,
                            num_samples=NUM_SAMPLES, progress_bar=True)
                mcmc.run(rng_key_current, data)
                samples = mcmc.get_samples()

                mu_s = np.array(samples['mu'])
                r_s  = np.array(samples['r'])
                print(f"  bin {b:2d}: n={n:7d}  "
                      f"mu={mu_s.mean():.3f}±{mu_s.std():.3f}  "
                      f"r={r_s.mean():.3f}±{r_s.std():.3f}")

                bin_fits.append({
                    'n'   : n,
                    'data': np.array(data),
                    'mu'  : float(mu_s.mean()),
                    'r'   : float(r_s.mean()),
                })
            fits[(cen_label, sat_col)] = bin_fits

    return fits


# ── PMF comparison grids ───────────────────────────────────────────────────────

def plot_pmf_grid(cen_label, sat_col, bin_fits, bin_cents, color, fname):
    """
    8×5 grid — one panel per mass bin.

    Empirical PMF : coloured bars, drawn only at k with count > 0
                    (skipping zeros avoids log(0) on the log y-axis).
    NB PMF        : solid black line over 0 … N_max.
    x-range       : 0 to N_max = ceil(mu + 7*sigma_NB).
    y-scale       : log; floor = 0.3 / n  so single-count events are visible.
    """
    sat_sym = (r'N_\mathrm{sat}^\mathrm{SF}' if 'SF' in sat_col
               else r'N_\mathrm{sat}^\mathrm{Q}')

    fig, axes = plt.subplots(8, 5, figsize=(16, 26))

    for idx, ax in enumerate(axes.flat):
        fit  = bin_fits[idx]
        logM = np.log10(bin_cents[idx])
        ax.set_title(rf'$\log M={logM:.2f}$', fontsize=7)
        ax.tick_params(labelsize=6)

        if fit is None:
            ax.text(0.5, 0.5, 'too few\nhalos',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='grey')
            continue

        mu, r = fit['mu'], fit['r']
        data  = fit['data']
        n     = fit['n']

        sigma = np.sqrt(mu + mu**2 / r)
        N_max = int(np.ceil(mu + 7. * sigma))
        k_arr = np.arange(0, N_max + 1)

        counts  = np.bincount(data, minlength=N_max + 1)[:N_max + 1]
        emp_pmf = counts / counts.sum()
        mask    = emp_pmf > 0

        # NegBinom2: mean=mu, conc=r → n=r, p=r/(r+mu) in scipy parameterisation
        nb_pmf = scipy_nbinom.pmf(k_arr, r, r / (r + mu))

        ax.bar(k_arr[mask], emp_pmf[mask], color=color, alpha=0.65,
               width=0.5, label='Data', zorder=2)
        ax.plot(k_arr, nb_pmf, color='k', ls='-', lw=1.2,
                label='NB fit', zorder=3)

        ax.set_yscale('log')
        ax.set_xlim(-0.5, N_max + 0.5)
        ax.set_ylim(bottom=0.3 / n)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.annotate(f'n={n}', xy=(0.97, 0.97), xycoords='axes fraction',
                    ha='right', va='top', fontsize=5.5)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    fig.suptitle(rf'PMF of ${sat_sym}$ — {cen_label} — Data vs NB fit',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(f'figs/exploration/negbinom_check/{fname}', dpi=150)
    plt.close()


def plot_summary(fits, subsets, bin_cents):
    """
    2 rows × 2 cols  (height_ratios [3, 1], sharex per column).
      Col 0 : N_sat_SF       Col 1 : N_sat_Q
      Row 0 : mean           Row 1 : Fano factor  (variance / mean)

    Colours  : steelblue = SF central,  firebrick = Q central
    Data     : solid line,  markers 'o' (SF cen) / 's' (Q cen)
    NB fit   : dashed line, markers '^' (SF cen) / 'D' (Q cen)
    x-offset : data at bin_cents*(1-eps), NB fit at bin_cents*(1+eps)
    """
    eps    = 0.04
    x_data = bin_cents * (1. - eps)
    x_fit  = bin_cents * (1. + eps)

    cen_props = {
        'SF central': dict(color='steelblue', mk_data='o', mk_fit='^'),
        'Q central' : dict(color='firebrick',  mk_data='s', mk_fit='D'),
    }
    sat_specs = [
        ('N_sat_SF', r'$N_\mathrm{sat}^\mathrm{SF}$'),
        ('N_sat_Q',  r'$N_\mathrm{sat}^\mathrm{Q}$'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [3, 1]},
                             sharex='col')

    for col_idx, (sat_col, sat_label) in enumerate(sat_specs):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        for cen_label, props in cen_props.items():
            color  = props['color']
            df     = subsets[cen_label]
            bf     = fits[(cen_label, sat_col)]

            grp       = df.groupby('mass_bin')[sat_col]
            mean_data = grp.mean().reindex(range(N_BINS)).values
            var_data  = grp.var(ddof=1).reindex(range(N_BINS)).values
            fano_data = var_data / mean_data

            mu_fit   = np.array([f['mu'] if f is not None else np.nan for f in bf])
            r_fit    = np.array([f['r']  if f is not None else np.nan for f in bf])
            fano_fit = 1. + mu_fit / r_fit

            ax_top.loglog(x_data, mean_data, color=color, ls='-',
                          marker=props['mk_data'], ms=4,
                          label=f'{cen_label} data')
            ax_top.loglog(x_fit, mu_fit, color=color, ls='--',
                          marker=props['mk_fit'], ms=4,
                          label=f'{cen_label} NB fit')

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

    fig.suptitle('HOD satellites — NB fit: Mean & Fano', fontsize=13)
    plt.tight_layout()
    plt.savefig('figs/exploration/negbinom_check/summary.png', dpi=150)
    plt.close()


def main():
    record_library_versions('figs/exploration/negbinom_check/library_versions.json')
    save_mcmc_run_config(
        'figs/exploration/negbinom_check/run_config.json',
        config=dict(
            script_name  = 'negbinom_check.py',
            num_warmup   = NUM_WARMUP,
            num_samples  = NUM_SAMPLES,
            prior_val    = PRIOR_VAL,
            rng_seed     = RNG_SEED,
            n_bins       = N_BINS,
            data_file    = DATA_FILE_FULL,
        ),
    )

    hod_df  = load_hod_dataframe(use_full=True)
    subsets = split_by_central_type(hod_df)

    fits = fit_negbinom_per_bin(subsets)

    for cen_label in ['SF central', 'Q central']:
        cen_tag = 'sf_cen' if 'SF' in cen_label else 'q_cen'
        for sat_col in ['N_sat_SF', 'N_sat_Q']:
            sat_tag = 'sf_sat' if 'SF' in sat_col else 'q_sat'
            color   = 'steelblue' if 'SF' in sat_col else 'firebrick'
            print(f"\nPlotting PMF grid: {cen_label} | {sat_col}")
            plot_pmf_grid(cen_label, sat_col,
                          fits[(cen_label, sat_col)],
                          BIN_CENTS, color,
                          fname=f'pmf_{sat_tag}_{cen_tag}.png')

    print("\nPlotting summary...")
    plot_summary(fits, subsets, BIN_CENTS)
    print("\nDone. Figures saved to figs/exploration/negbinom_check/")


if __name__ == '__main__':
    main()
