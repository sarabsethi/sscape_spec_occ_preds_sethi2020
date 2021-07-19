from pc_data_tools import load_pc_dataset, get_avi_pcs_no_water_sites, get_herp_pcs_no_water_sites
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.03, 0.7])

data_fname = 'pc_data_parsed_raw_audioset_feats_3s'
data_path = os.path.join('pc_data', data_fname)

feat_name, all_sites, all_taxa, all_pcs = load_pc_dataset(data_fname)

avi_pcs_no_water = np.asarray(get_avi_pcs_no_water_sites(all_pcs, all_sites))
herp_pcs_no_water = np.asarray(get_herp_pcs_no_water_sites(all_pcs, all_sites))
print('Total PCs: {} avi {} herp'.format(len(avi_pcs_no_water), len(herp_pcs_no_water)))

titles = ['Avifaunal point count sampling effort', 'Herpetofaunal point count sampling effort']

for pc_list, ax, title in zip([avi_pcs_no_water, herp_pcs_no_water], [ax1, ax2], titles):

    pc_site_list = np.asarray([pc.site for pc in pc_list])

    _, unq_idx = np.unique([s.name for s in pc_site_list], return_index=True)

    unq_sites = pc_site_list[unq_idx]
    sort_ix = np.argsort([s.get_agb() for s in unq_sites])
    unq_sites = unq_sites[sort_ix]

    avi_pc_ids = np.where([pc.id for pc in pc_list])
    pc_site_names = np.asarray([pc.site.name for pc in pc_list])

    sampling_effort_mat = []

    for s in unq_sites:
        site_ixs = np.where((pc_site_names == s.name))[0]
        #print('{}: {} matching PCs'.format(s.name, len(site_ixs)))

        site_pcs = pc_list[site_ixs]
        site_pc_hrs = np.asarray([pc.dt.hour for pc in site_pcs])

        hr_effort_vec = []
        for hr in range(0,24):
            hr_ixs = np.where((site_pc_hrs == hr))[0]
            #print('{}: {} matching hours'.format(hr, len(hr_ixs)))
            hr_effort_vec.append(len(hr_ixs))

        sampling_effort_mat.append(hr_effort_vec)

    sampling_effort_mat = np.asarray(sampling_effort_mat)
    print('{} sampling effort: mean = {}, std = {}'.format(title, np.mean(sampling_effort_mat), np.std(sampling_effort_mat)))

    plt.sca(ax)
    cmap_ticks = 7
    cmap = plt.get_cmap('viridis', cmap_ticks)

    im = plt.matshow(sampling_effort_mat, cmap=cmap, fignum=False, vmin=0, vmax=6)
    plt.gca().set_xticks(np.arange(24))
    plt.gca().set_xticklabels(np.arange(24))
    plt.gca().set_yticks(np.arange(len(unq_sites)))
    plt.gca().set_yticklabels(['{} ({})'.format(s.get_abbrv_name(), s.name) for s in unq_sites])

    plt.xlabel('Hour of day')
    plt.ylabel('Site')

    plt.title(title)

cbar = fig.colorbar(im, cax=cbar_ax)
tick_locs = (np.arange(cmap_ticks) + 0.5)*(cmap_ticks-1)/cmap_ticks
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(np.arange(cmap_ticks))
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel('Number of point counts', rotation=270)

plt.tight_layout()

plt.savefig(os.path.join('figs','fig_sampling_effort.pdf'),format='pdf')
plt.show()
