import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('figs/exploration', exist_ok=True)

# hod_df = pd.read_csv('./data/hod_z0.csv')
hod_df = pd.read_csv('./data/hod_z0_small.csv')

# ── mass bins ─────────────────────────────────────────────────────────────────
MASS_COL  = 'halo_mp'
log_edges = np.linspace(11., 15., 41)
edges     = 10**log_edges
bin_cents = 10**((log_edges[:-1] + log_edges[1:]) / 2)

hod_df['mass_bin'] = pd.cut(hod_df[MASS_COL], bins=edges, labels=False)

# ── subsets by central type ───────────────────────────────────────────────────
df_all    = hod_df.copy()
df_sf_cen = hod_df[hod_df['N_cen_SF'] == 1].copy()
df_q_cen  = hod_df[hod_df['N_cen_Q']  == 1].copy()

subsets = {
    'All'        : df_all,
    'SF central' : df_sf_cen,
    'Q central'  : df_q_cen,
}

central_style = {
    'SF central' : ('-',  'o'),
    'Q central'  : ('--', 's'),
}

color_map = {
    'SF central' : 'steelblue',
    'Q central'  : 'firebrick',
}


def plot_halo_counts(subsets, bin_cents, central_style):
    """
    Number of halos in each mass bin split by central type (SF / Q).
    Top panel  : raw counts with Poisson error bars sqrt(N).
    Bottom panel: fraction of total with propagated Poisson errors.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8),
                                          gridspec_kw={'height_ratios': [3, 1]},
                                          sharex=True)

    counts = {}
    for cen_label in central_style:
        counts[cen_label] = (subsets[cen_label]
                             .groupby('mass_bin')
                             .size()
                             .reindex(range(len(bin_cents)), fill_value=0)
                             .values.astype(float))

    total = sum(counts.values())

    ax_top.set_xlim(1e+11, 1e+15)
    ax_bot.set_xlim(1e+11, 1e+15)
    
    # ── top panel: raw counts with Poisson errors ─────────────────────────────
    for cen_label, (ls, mk) in central_style.items():
        c   = counts[cen_label]
        err = np.sqrt(c)
        ax_top.errorbar(bin_cents, c, yerr=err,
                        color=color_map[cen_label], ls=ls, marker=mk, ms=4,
                        capsize=2, lw=1, label=cen_label)

    ax_top.set_xscale('log')
    ax_top.set_yscale('log')
    ax_top.set_ylabel('Number of halos')
    ax_top.set_title('Halo counts by central type')
    ax_top.legend(fontsize=9, title='Central type')
    ax_top.grid(True, which='both', ls=':', alpha=0.4)

    # ── bottom panel: fractions with propagated Poisson errors ────────────────
    for cen_label, (ls, mk) in central_style.items():
        c   = counts[cen_label]
        f   = np.where(total > 0, c / total, np.nan)
        # sigma_f = f * sqrt(1/c + 1/total)  (Poisson propagation)
        err = f * np.sqrt(np.where(c > 0, 1. / c, 0.)
                          + np.where(total > 0, 1. / total, 0.))
        ax_bot.errorbar(bin_cents, f, yerr=err,
                        color=color_map[cen_label], ls=ls, marker=mk, ms=4,
                        capsize=2, lw=1, label=cen_label)

    ax_bot.axhline(1, color='k', ls='--', lw=0.8, alpha=0.4)
    ax_bot.set_xscale('log')
    ax_bot.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
    ax_bot.set_ylabel('Fraction of total')
    ax_bot.legend(fontsize=8, title='Central type')
    ax_bot.grid(True, which='both', ls=':', alpha=0.4)

    fig.suptitle('Halo counts by central type vs halo mass', fontsize=13)
    plt.tight_layout()
    plt.savefig('figs/exploration/halo_counts.png', dpi=150)
    plt.close()


# ── helpers ───────────────────────────────────────────────────────────────────
def bin_stats(df, col):
    """
    Per-bin mean, Fano factor, and their uncertainties.

    Returns
    -------
    mean, fano, sem_mean, sem_fano, n   (all length-n_bins arrays)

    sem_mean = std / sqrt(n)
    sem_fano = Fano * sqrt(2 / (n-1))   [delta-method, normal approx on s²]
    """
    grp      = df.groupby('mass_bin')[col]
    mean     = grp.mean().reindex(range(len(bin_cents)))
    var      = grp.var(ddof=1).reindex(range(len(bin_cents)))
    n        = grp.count().reindex(range(len(bin_cents)), fill_value=0)
    fano     = var / mean
    sem_mean = np.sqrt(var) / np.sqrt(n)
    sem_fano = fano * np.sqrt(2. / (n - 1).clip(lower=1))
    return (mean.values, fano.values,
            sem_mean.values, sem_fano.values, n.values.astype(float))


def bin_corr(df, col_x, col_y):
    """
    Per-bin Pearson r with asymmetric Fisher-z error bars.

    Returns r, err_lo, err_hi   (all length-n_bins arrays).
    err_lo = r - tanh(arctanh(r) - 1/sqrt(n-3))
    err_hi = tanh(arctanh(r) + 1/sqrt(n-3)) - r
    """
    grp  = df.groupby('mass_bin')
    r    = (grp.apply(lambda g: g[col_x].corr(g[col_y]))
               .reindex(range(len(bin_cents))).values)
    n    = (grp.size()
               .reindex(range(len(bin_cents)), fill_value=0).values.astype(float))
    z    = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se_z = 1. / np.sqrt(np.maximum(n - 3., 1.))
    err_lo = r - np.tanh(z - se_z)
    err_hi = np.tanh(z + se_z) - r
    return r, err_lo, err_hi


def plot_mean_occupation(subsets, bin_cents, central_style):
    """
    Row 1, Panel 1 : mean N_cen^SF and N_cen^Q (all halos).
    Row 1, Panels 2-3 : mean N_sat^SF / N_sat^Q split by central type.
    Row 2, Panels 2-3 : ratio Q-central / SF-central with propagated SEM errors.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 7),
                             gridspec_kw={'height_ratios': [3, 1]})

    # ── top-left: centrals (all halos) ───────────────────────────────────────
    axes[1, 0].set_visible(False)
    ax = axes[0, 0]
    mean_cen_sf, _, sem_sf, _, _ = bin_stats(subsets['All'], 'N_cen_SF')
    mean_cen_q,  _, sem_q,  _, _ = bin_stats(subsets['All'], 'N_cen_Q')
    ax.errorbar(bin_cents, mean_cen_sf, yerr=sem_sf,
                color='steelblue', ls='-', marker='o', ms=4, capsize=2, lw=1,
                label=r'$\langle N_\mathrm{cen}^\mathrm{SF} \rangle$')
    ax.errorbar(bin_cents, mean_cen_q,  yerr=sem_q,
                color='firebrick',  ls='-', marker='o', ms=4, capsize=2, lw=1,
                label=r'$\langle N_\mathrm{cen}^\mathrm{Q} \rangle$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1.)
    ax.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
    ax.set_ylabel('Mean occupation')
    ax.set_title('Centrals (all halos)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.4)

    # ── satellite panels ──────────────────────────────────────────────────────
    sat_specs = [
        ('N_sat_SF', 'steelblue', r'SF satellites ($N_\mathrm{sat}^\mathrm{SF}$)', 1),
        ('N_sat_Q',  'firebrick', r'Q satellites ($N_\mathrm{sat}^\mathrm{Q}$)',   2),
    ]

    for sat_col, sat_color, sat_title, col_idx in sat_specs:
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        means = {}; sems = {}
        for cen_label, (ls, mk) in central_style.items():
            mean, _, sem, _, _ = bin_stats(subsets[cen_label], sat_col)
            means[cen_label] = mean
            sems[cen_label]  = sem
            ax_top.errorbar(bin_cents, mean, yerr=sem,
                            color=sat_color, ls=ls, marker=mk, ms=4,
                            capsize=2, lw=1, label=cen_label)

        ax_top.set_xscale('log')
        ax_top.set_yscale('log')
        ax_top.set_ylabel('Mean occupation')
        ax_top.set_title(sat_title)
        ax_top.legend(fontsize=9, title='Central type')
        ax_top.grid(True, which='both', ls=':', alpha=0.4)
        ax_top.set_xticklabels([])

        # ── ratio panel with propagated errors ────────────────────────────────
        ref     = means['SF central']
        ref_sem = sems['SF central']
        cen_label, (ls, mk) = 'Q central', central_style['Q central']
        ratio = means[cen_label] / ref
        # SE(A/B) = (A/B) * sqrt((SE_A/A)^2 + (SE_B/B)^2)
        err   = ratio * np.sqrt(
            (sems[cen_label] / np.where(means[cen_label] > 0, means[cen_label], np.nan))**2
            + (ref_sem / np.where(ref > 0, ref, np.nan))**2)
        ax_bot.errorbar(bin_cents, ratio, yerr=err,
                        color=sat_color, ls=ls, marker=mk, ms=4,
                        capsize=2, lw=1, label=cen_label)

        ax_bot.set_ylim(0.5, 1.5)
        ax_bot.axhline(1, color='k', ls='--', lw=0.8, label='SF central (ref)')
        ax_bot.set_xscale('log')
        ax_bot.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
        ax_bot.set_ylabel('Ratio to SF cen')
        ax_bot.legend(fontsize=8, title='Central type')
        ax_bot.grid(True, which='both', ls=':', alpha=0.4)

    fig.suptitle('HOD — Mean occupation', fontsize=13)
    plt.tight_layout()
    plt.savefig('figs/exploration/mean_occupation.png', dpi=150)
    plt.close()


def plot_fano(subsets, bin_cents, central_style):
    """
    Row 1, Panel 1 : Fano factor for N_cen^SF and N_cen^Q (all halos).
    Row 1, Panels 2-3 : Fano for N_sat^SF / N_sat^Q split by central type.
    Row 2, Panels 2-3 : ratio Q-central / SF-central Fano with propagated errors.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 7),
                             gridspec_kw={'height_ratios': [3, 1]})

    # ── top-left: centrals ────────────────────────────────────────────────────
    axes[1, 0].set_visible(False)
    ax = axes[0, 0]
    _, fano_cen_sf, _, sem_fano_sf, _ = bin_stats(subsets['All'], 'N_cen_SF')
    _, fano_cen_q,  _, sem_fano_q,  _ = bin_stats(subsets['All'], 'N_cen_Q')
    ax.errorbar(bin_cents, fano_cen_sf, yerr=sem_fano_sf,
                color='steelblue', ls='-', marker='o', ms=4, capsize=2, lw=1,
                label=r'$N_\mathrm{cen}^\mathrm{SF}$')
    ax.errorbar(bin_cents, fano_cen_q,  yerr=sem_fano_q,
                color='firebrick',  ls='-', marker='o', ms=4, capsize=2, lw=1,
                label=r'$N_\mathrm{cen}^\mathrm{Q}$')
    ax.axhline(1, color='k', ls='--', lw=0.8, label='Poisson (=1)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
    ax.set_ylabel('Variance / Mean')
    ax.set_title('Centrals (all halos)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.4)

    # ── satellite panels ──────────────────────────────────────────────────────
    sat_specs = [
        ('N_sat_SF', 'steelblue', r'SF satellites ($N_\mathrm{sat}^\mathrm{SF}$)', 1),
        ('N_sat_Q',  'firebrick', r'Q satellites ($N_\mathrm{sat}^\mathrm{Q}$)',   2),
    ]

    for sat_col, sat_color, sat_title, col_idx in sat_specs:
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        fanos = {}; sem_fanos = {}
        for cen_label, (ls, mk) in central_style.items():
            _, fano, _, sem_fano, _ = bin_stats(subsets[cen_label], sat_col)
            fanos[cen_label]     = fano
            sem_fanos[cen_label] = sem_fano
            ax_top.errorbar(bin_cents, fano, yerr=sem_fano,
                            color=sat_color, ls=ls, marker=mk, ms=4,
                            capsize=2, lw=1, label=cen_label)

        ax_top.axhline(1, color='k', ls='--', lw=0.8, label='Poisson (=1)')
        ax_top.set_ylim(0., 5.)
        ax_top.set_xscale('log')
        ax_top.set_ylabel('Variance / Mean')
        ax_top.set_title(sat_title)
        ax_top.legend(fontsize=9, title='Central type')
        ax_top.grid(True, which='both', ls=':', alpha=0.4)
        ax_top.set_xticklabels([])

        # ── ratio panel with propagated errors ────────────────────────────────
        ref     = fanos['SF central']
        ref_sem = sem_fanos['SF central']
        cen_label, (ls, mk) = 'Q central', central_style['Q central']
        ratio = fanos[cen_label] / ref
        err   = ratio * np.sqrt(
            (sem_fanos[cen_label] / np.where(fanos[cen_label] > 0, fanos[cen_label], np.nan))**2
            + (ref_sem / np.where(ref > 0, ref, np.nan))**2)
        ax_bot.errorbar(bin_cents, ratio, yerr=err,
                        color=sat_color, ls=ls, marker=mk, ms=4,
                        capsize=2, lw=1, label=cen_label)

        ax_bot.axhline(1, color='k', ls='--', lw=0.8, label='SF central (ref)')
        ax_bot.set_ylim(0.5, 1.5)
        ax_bot.set_xscale('log')
        ax_bot.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
        ax_bot.set_ylabel('Ratio to SF cen')
        ax_bot.legend(fontsize=8, title='Central type')
        ax_bot.grid(True, which='both', ls=':', alpha=0.4)

    fig.suptitle('HOD — Fano factor (variance/mean)', fontsize=13)
    plt.tight_layout()
    plt.savefig('figs/exploration/fano_factor.png', dpi=150)
    plt.close()


def plot_cross_correlation(subsets, bin_cents, central_style):
    """
    Pearson r between N_sat^SF and N_sat^Q split by central type.
    Asymmetric error bars from Fisher z-transform: SE_z = 1/sqrt(n-3).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='k', ls=':', lw=0.8)

    for cen_label, (ls, mk) in central_style.items():
        r, err_lo, err_hi = bin_corr(subsets[cen_label], 'N_sat_SF', 'N_sat_Q')
        ax.errorbar(bin_cents, r, yerr=[err_lo, err_hi],
                    color=color_map[cen_label], ls=ls, marker=mk, ms=4,
                    capsize=2, lw=1, alpha=0.85, label=cen_label)

    ax.set_xscale('log')
    ax.set_xlabel(r'$M_\mathrm{peak}\ [M_\odot/h]$')
    ax.set_ylabel(r'Pearson $r$')
    ax.set_title(r'Cross-correlation: $N_\mathrm{sat}^\mathrm{SF}$ vs $N_\mathrm{sat}^\mathrm{Q}$')
    ax.legend(fontsize=9, title='Central type')
    ax.grid(True, which='both', ls=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig('figs/exploration/cross_correlation.png', dpi=150)
    plt.close()


# ── run ───────────────────────────────────────────────────────────────────────
print("Plotting halo counts....")
plot_halo_counts(subsets, bin_cents, central_style)
print("Plotting mean functions....")
plot_mean_occupation(subsets, bin_cents, central_style)
print("Plotting variances....")
plot_fano(subsets, bin_cents, central_style)
print("Plotting cross-correlations....")
plot_cross_correlation(subsets, bin_cents, central_style)
