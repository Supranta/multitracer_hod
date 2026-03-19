import numpy as np
import pandas as pd


def main():
    # ── dtype & load ──────────────────────────────────────────────────────────
    dtype = np.dtype(dtype=[('id', 'i8'), ('descid', 'i8'), ('upid', 'i8'),
                            ('flags', 'i4'), ('uparent_dist', 'f4'),
                            ('pos', 'f4', (6)), ('vmp', 'f4'), ('lvmp', 'f4'),
                            ('mp', 'f4'), ('m', 'f4'), ('v', 'f4'), ('r', 'f4'),
                            ('rank1', 'f4'), ('rank2', 'f4'), ('ra', 'f4'),
                            ('rarank', 'f4'), ('A_UV', 'f4'), ('sm', 'f4'),
                            ('icl', 'f4'), ('sfr', 'f4'), ('obs_sm', 'f4'),
                            ('obs_sfr', 'f4'), ('obs_uv', 'f4'), ('empty', 'f4')],
                     align=True)

    datafile    = '../MBI/data/UniverseMachine/sfr_catalog_1.000000.bin'
    print("Loading UM galaxies....")
    um_galaxies = np.fromfile(datafile, dtype=dtype)
    assert len(um_galaxies) > 0, f"No galaxies loaded from {datafile}"

    # ── thresholds & SF/Q flags ───────────────────────────────────────────────
    # Halo peak mass > 10^11 M_sun/h and stellar mass > 10^9.2 M_sun/h are the
    # completeness cuts for this UniverseMachine snapshot (z=0).
    mass_threshold     = 1e+11
    sm_threshold       = 10**9.2
    QUENCHED_THRESHOLD = -11.    # log10(sSFR / yr^-1) boundary

    SM      = um_galaxies['sm']
    sfr     = um_galaxies['sfr']
    ssfr    = sfr / SM
    m       = um_galaxies['m']
    mp      = um_galaxies['mp']
    halo_id = um_galaxies['id']
    upid    = um_galaxies['upid']

    del um_galaxies

    select_mass_thresholds = (mp > mass_threshold) & (SM > sm_threshold)
    select_quenched        = select_mass_thresholds & (np.log10(ssfr) < QUENCHED_THRESHOLD)
    select_starforming     = select_mass_thresholds & (np.log10(ssfr) > QUENCHED_THRESHOLD)

    # ── separate centrals (upid == -1) from satellites (upid != -1) ──────────
    is_central   = upid == -1
    is_satellite = upid != -1

    # Index by halo_id (the central's own ID), NOT upid (which is -1 for all
    # centrals), so satellites can be matched to their host via upid lookup.
    print("Populating pandas df with central galaxies....")
    hod_df = pd.DataFrame({
        'halo_id' : halo_id[is_central],
        'halo_m'  : m[is_central],
        'halo_mp' : mp[is_central],
        'N_cen_SF': np.where(select_starforming[is_central], 1, 0).astype(np.int32),
        'N_cen_Q' : np.where(select_quenched[is_central],   1, 0).astype(np.int32),
        'N_sat_SF': np.zeros(is_central.sum(), dtype=np.int32),
        'N_sat_Q' : np.zeros(is_central.sum(), dtype=np.int32),
    }).set_index('halo_id')

    # ── count satellites per host halo ────────────────────────────────────────
    print("Selecting satellite galaxies....")
    sat_upids   = upid[is_satellite]
    sat_sf_mask = select_starforming[is_satellite]
    sat_q_mask  = select_quenched[is_satellite]

    sat_any   = sat_sf_mask | sat_q_mask
    sat_upids = sat_upids[sat_any]
    sat_is_sf = sat_sf_mask[sat_any]
    sat_is_q  = sat_q_mask[sat_any]

    print("Counting satellite galaxies per unique host halo ID....")
    sf_counts = (pd.Series(sat_upids[sat_is_sf])
                   .value_counts()
                   .reindex(hod_df.index, fill_value=0))
    q_counts  = (pd.Series(sat_upids[sat_is_q])
                   .value_counts()
                   .reindex(hod_df.index, fill_value=0))

    print("Adding satellite counts to the dataframe....")
    hod_df['N_sat_SF'] += sf_counts.values
    hod_df['N_sat_Q']  += q_counts.values

    del SM, sfr, ssfr, m, mp, halo_id, upid
    del select_mass_thresholds, select_quenched, select_starforming
    del is_central, is_satellite
    del sat_upids, sat_sf_mask, sat_q_mask, sat_any, sat_is_sf, sat_is_q
    del sf_counts, q_counts

    # ── total galaxy count & filter ───────────────────────────────────────────
    hod_df['N_tot'] = (hod_df['N_cen_SF'] + hod_df['N_cen_Q']
                       + hod_df['N_sat_SF'] + hod_df['N_sat_Q'])

    n_before = len(hod_df)
    hod_df   = hod_df[hod_df['N_tot'] >= 1]
    n_after  = len(hod_df)
    print(f"Removed {n_before - n_after:,} halos with no qualifying galaxies "
          f"({n_after:,} remaining).")

    expected_columns = {'halo_m', 'halo_mp', 'N_cen_SF', 'N_cen_Q', 'N_sat_SF', 'N_sat_Q'}
    assert expected_columns <= set(hod_df.columns), "Missing expected columns in hod_df"
    assert len(hod_df) > 0, "hod_df is empty after filtering"

    hod_df_small = hod_df.sample(frac=0.1, random_state=42)

    hod_df.to_csv('./data/hod_z0.csv')
    hod_df_small.to_csv('./data/hod_z0_small.csv', index=False)
    print("Done.")


if __name__ == '__main__':
    main()
