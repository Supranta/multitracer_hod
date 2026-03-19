import numpy as np
import pandas as pd

# ── dtype & load ──────────────────────────────────────────────────────────────
dtype = np.dtype(dtype=[('id', 'i8'),('descid','i8'),('upid','i8'),
                        ('flags', 'i4'), ('uparent_dist', 'f4'),
                        ('pos', 'f4', (6)), ('vmp', 'f4'), ('lvmp', 'f4'),
                        ('mp', 'f4'), ('m', 'f4'), ('v', 'f4'), ('r', 'f4'),
                        ('rank1', 'f4'), ('rank2', 'f4'), ('ra', 'f4'),
                        ('rarank', 'f4'), ('A_UV', 'f4'), ('sm', 'f4'),
                        ('icl', 'f4'), ('sfr', 'f4'), ('obs_sm', 'f4'),
                        ('obs_sfr', 'f4'), ('obs_uv', 'f4'), ('empty', 'f4')],
                 align=True)

print("Loading UM galaxies....")
datafile    = '../MBI/data/UniverseMachine/sfr_catalog_1.000000.bin'
um_galaxies = np.fromfile(datafile, dtype=dtype)

# ── thresholds & SF/Q flags ───────────────────────────────────────────────────
mass_threshold     = 1e+11
sm_threshold       = 10**9.2
QUENCHED_THRESHOLD = -11.

SM      = um_galaxies['sm']
sfr     = um_galaxies['sfr']
ssfr    = sfr / SM
m       = um_galaxies['m']
mp      = um_galaxies['mp']
halo_id = um_galaxies['id']
upid    = um_galaxies['upid']

print("Deleting um_galaxies....")
del um_galaxies

select_mass_thresholds = (mp > mass_threshold) & (SM > sm_threshold)
select_quenched        = select_mass_thresholds & (np.log10(ssfr) < QUENCHED_THRESHOLD)
select_starforming     = select_mass_thresholds & (np.log10(ssfr) > QUENCHED_THRESHOLD)

# ── separate centrals (upid == -1) from satellites (upid != -1) ───────────────
is_central   = upid == -1
is_satellite = upid != -1

print("Populating pandas df with central galaxies....")
# Index by halo_id (the central's own ID), NOT upid (which is -1 for all centrals)
hod_df = pd.DataFrame({
    'halo_id' : halo_id[is_central],      # ← fix: use halo_id, not upid
    'halo_m'  : m[is_central],
    'halo_mp' : mp[is_central],
    'N_cen_SF': np.where(select_starforming[is_central], 1, 0).astype(np.int32),
    'N_cen_Q' : np.where(select_quenched[is_central],   1, 0).astype(np.int32),
    'N_sat_SF': np.zeros(is_central.sum(), dtype=np.int32),
    'N_sat_Q' : np.zeros(is_central.sum(), dtype=np.int32),
}).set_index('halo_id')

print("Selecting satellite galaxies....")
# Step 1 — pull the upids (= parent host halo ID) and SF/Q masks for satellites only
sat_upids  = upid[is_satellite]       # parent host halo ID for every satellite
sat_sf_mask = select_starforming[is_satellite]
sat_q_mask  = select_quenched[is_satellite]

# Step 2 — restrict to satellites that pass at least one threshold
sat_any    = sat_sf_mask | sat_q_mask
sat_upids  = sat_upids[sat_any]
sat_is_sf  = sat_sf_mask[sat_any]
sat_is_q   = sat_q_mask[sat_any]

print("Counting satellite galaxies per unique host halo ID....")
# Step 3 — for each type, get the unique parent halo IDs and count occurrences,
#           then reindex against hod_df (which is indexed by host halo_id)
sf_counts = (pd.Series(sat_upids[sat_is_sf])   # upids of SF satellites
               .value_counts()                  # {host_halo_id: count}
               .reindex(hod_df.index, fill_value=0))

q_counts  = (pd.Series(sat_upids[sat_is_q])    # upids of Q satellites
               .value_counts()
               .reindex(hod_df.index, fill_value=0))

print("Adding satellite counts to the dataframe....")
hod_df['N_sat_SF'] += sf_counts.values
hod_df['N_sat_Q']  += q_counts.values

print("Deleting unnecessary variables....")
del SM, sfr, ssfr, m, mp, halo_id, upid
del select_mass_thresholds, select_quenched, select_starforming
del is_central, is_satellite
del sat_upids, sat_sf_mask, sat_q_mask, sat_any, sat_is_sf, sat_is_q
del sf_counts, q_counts

# ── total galaxy count & filter ───────────────────────────────────────────────
hod_df['N_tot'] = (hod_df['N_cen_SF'] + hod_df['N_cen_Q']
                   + hod_df['N_sat_SF'] + hod_df['N_sat_Q'])

n_before = len(hod_df)
hod_df   = hod_df[hod_df['N_tot'] >= 1]
n_after  = len(hod_df)
print(f"Removed {n_before - n_after:,} halos with no qualifying galaxies "
      f"({n_after:,} remaining).")

print("Done.")

hod_df_small = hod_df.sample(frac=0.1, random_state=42)

hod_df.to_csv('./data/hod_z0.csv')
hod_df_small.to_csv('./data/hod_z0_small.csv', index=False)