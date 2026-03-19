import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('figs/exploration/bernoulli_check', exist_ok=True)

# ── load & bin (identical setup to data_exploration.py) ───────────────────────
# hod_df    = pd.read_csv('./data/hod_z0.csv')
hod_df = pd.read_csv('./data/hod_z0_small.csv')

MASS_COL  = 'halo_mp'
log_edges = np.linspace(11., 15., 41)
edges     = 10**log_edges
bin_cents = 10**((log_edges[:-1] + log_edges[1:]) / 2)
n_bins    = len(bin_cents)   # 40

hod_df['mass_bin'] = pd.cut(hod_df[MASS_COL], bins=edges, labels=False)

# ── per-bin: empirical means and a single Bernoulli draw ──────────────────────
rng     = np.random.default_rng(42)
results = []

for b in range(n_bins):
    sub = hod_df[hod_df['mass_bin'] == b]
    n   = len(sub)

    data_sf = sub['N_cen_SF'].values
    data_q  = sub['N_cen_Q'].values

    p_sf = data_sf.mean() if n > 0 else np.nan
    p_q  = data_q.mean()  if n > 0 else np.nan

    draw_sf = rng.binomial(1, p_sf, size=n) if (n > 0 and not np.isnan(p_sf)) else np.array([])
    draw_q  = rng.binomial(1, p_q,  size=n) if (n > 0 and not np.isnan(p_q))  else np.array([])

    results.append({
        'n'      : n,
        'p_sf'   : p_sf,
        'p_q'    : p_q,
        'data_sf': data_sf,
        'data_q' : data_q,
        'draw_sf': draw_sf,
        'draw_q' : draw_q,
    })

# ── PMF comparison grid ────────────────────────────────────────────────────────
def plot_pmf_grid(results, bin_cents, data_key, draw_key, color, title, fname):
    """
    8×5 grid (40 panels, one per mass bin). Each panel shows the 2-point PMF
    (P(0) and P(1)) for the data vs a single Bernoulli draw in that mass bin.
    """
    nrows, ncols = 8, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 24))

    for idx, ax in enumerate(axes.flat):
        if idx >= n_bins:
            ax.set_visible(False)
            continue

        r    = results[idx]
        data = r[data_key]
        draw = r[draw_key]

        p1_data = data.mean() if len(data) > 0 else np.nan
        p1_draw = draw.mean() if len(draw) > 0 else np.nan

        x     = np.array([0, 1])
        width = 0.35

        ax.bar(x - width/2, [1 - p1_data, p1_data], width,
               color=color, alpha=0.85, label='Data')
        ax.bar(x + width/2, [1 - p1_draw, p1_draw], width,
               color=color, alpha=0.4, hatch='//', label='Bernoulli draw')

        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.15)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'], fontsize=7)
        ax.set_title(rf'$\log M={np.log10(bin_cents[idx]):.2f}$'
                     rf'  (n={r["n"]})', fontsize=6.5)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(f'figs/exploration/bernoulli_check/{fname}', dpi=150)
    plt.close()


plot_pmf_grid(results, bin_cents,
              data_key='data_sf', draw_key='draw_sf',
              color='steelblue',
              title=r'PMF: $N_\mathrm{cen}^\mathrm{SF}$ — Data vs Bernoulli draw',
              fname='pmf_cen_sf.png')

plot_pmf_grid(results, bin_cents,
              data_key='data_q', draw_key='draw_q',
              color='firebrick',
              title=r'PMF: $N_\mathrm{cen}^\mathrm{Q}$ — Data vs Bernoulli draw',
              fname='pmf_cen_q.png')

# ── summary plot: mean & Fano factor ─────────────────────────────────────────
def plot_summary(results, bin_cents):
    """
    Top panel  : mean N_cen^SF and N_cen^Q vs mass.
                 Data (solid) and Bernoulli (dashed) are identical by
                 construction (p is estimated from the data) — sanity check.
    Bottom panel: Fano factor (variance/mean) vs mass.
                  Data (solid) vs Bernoulli prediction 1-p (dashed).
                  Deviation here indicates non-Bernoulli behaviour.
    """
    mean_sf = np.array([r['p_sf'] for r in results])
    mean_q  = np.array([r['p_q']  for r in results])

    # empirical Fano: Var(N_cen) / Mean(N_cen)
    def fano(data, p):
        if len(data) > 1 and p > 0:
            return np.var(data, ddof=1) / p
        return np.nan

    fano_sf_data = np.array([fano(r['data_sf'], r['p_sf']) for r in results])
    fano_q_data  = np.array([fano(r['data_q'],  r['p_q'])  for r in results])

    # Bernoulli prediction: Fano = Var / Mean = p(1-p) / p = 1 - p
    fano_sf_bern = 1. - mean_sf
    fano_q_bern  = 1. - mean_q

    # x-offsets so data and Bernoulli markers don't overlap (log-scale: multiply)
    eps      = 0.04
    x_data   = bin_cents * (1. - eps)
    x_bern   = bin_cents * (1. + eps)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8),
                                          gridspec_kw={'height_ratios': [3, 1]},
                                          sharex=True)

    # top: mean (data == Bernoulli by construction, shown as sanity check)
    ax_top.loglog(x_data, mean_sf, color='steelblue', ls='-',  marker='o', ms=4,
                  label=r'$\langle N_\mathrm{cen}^\mathrm{SF}\rangle$ data')
    ax_top.loglog(x_data, mean_q,  color='firebrick',  ls='-',  marker='s', ms=4,
                  label=r'$\langle N_\mathrm{cen}^\mathrm{Q}\rangle$ data')
    ax_top.loglog(x_bern, mean_sf, color='steelblue', ls='--', marker='^', ms=4,
                  label=r'Bernoulli SF')
    ax_top.loglog(x_bern, mean_q,  color='firebrick',  ls='--', marker='D', ms=4,
                  label=r'Bernoulli Q')
    ax_top.set_ylabel('Mean occupation')
    ax_top.set_title('Central galaxies: Data vs Bernoulli')
    ax_top.legend(fontsize=9)
    ax_top.grid(True, which='both', ls=':', alpha=0.4)

    # bottom: Fano factor
    ax_bot.semilogx(x_data, fano_sf_data, color='steelblue', ls='-',  marker='o', ms=4,
                    label=r'$N_\mathrm{cen}^\mathrm{SF}$ data')
    ax_bot.semilogx(x_data, fano_q_data,  color='firebrick',  ls='-',  marker='s', ms=4,
                    label=r'$N_\mathrm{cen}^\mathrm{Q}$ data')
    ax_bot.semilogx(x_bern, fano_sf_bern, color='steelblue', ls='--', marker='^', ms=4,
                    label=r'Bernoulli SF: $1-p$')
    ax_bot.semilogx(x_bern, fano_q_bern,  color='firebrick',  ls='--', marker='D', ms=4,
                    label=r'Bernoulli Q: $1-p$')
    ax_bot.axhline(1, color='k', ls=':', lw=0.8, label='Poisson (=1)')
    ax_bot.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
    ax_bot.set_ylabel('Variance / Mean')
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, which='both', ls=':', alpha=0.4)

    fig.suptitle('HOD centrals — Bernoulli check', fontsize=13)
    plt.tight_layout()
    plt.savefig('figs/exploration/bernoulli_check/summary.png', dpi=150)
    plt.close()


plot_summary(results, bin_cents)

print("Done. Figures saved to figs/exploration/bernoulli_check/")
