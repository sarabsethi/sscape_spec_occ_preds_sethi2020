import pickle
import os
import numpy as np
from pc_data_tools import get_nice_lab, get_spec_type_col
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
from matplotlib.lines import Line2D

def plot_auc_by_site_fig(all_spec_types, score_type='p60', feat='raw_audioset_feats_3s', results_dir='fig_data_logo'):
    '''
    Plot AUC for each species at each site as a scatter plot
    '''

    auc_per_site = None
    site_names = None

    spec_types = []
    leg_elements = []

    for spec_type in all_spec_types:
        # Load classification results for this species type
        f = 'classif_scores_sorted_{}_{}_{}.pickle'.format(feat,score_type,spec_type)
        load_path = os.path.join(results_dir, f)
        with open(load_path, 'rb') as f:
            auc_specs, spec_n_occs, _, aucs, auc_ks, _, _, _, _, _, all_k_sites = pickle.load(f)

        # Make sure all site names are in the same order across K folds
        if site_names is None:
            site_names = [s.name for s in all_k_sites[0]]
            site_agbs = [s.get_agb() for s in all_k_sites[0]]
            site_sort_ix_agb = np.argsort(site_agbs)
        else:
            for k_sites in all_k_sites:
                snames = [s.name for s in k_sites]
                assert(snames == site_names)

        # Append AUC results to an nparray storing results for all species types
        auc_per_site_st = np.asarray(auc_ks)
        if auc_per_site is None:
            auc_per_site = auc_per_site_st
        else:
            auc_per_site = np.vstack([auc_per_site, auc_per_site_st])

        # Track which rows correspond to which species type (for colouring points later)
        spec_types.extend([spec_type] * auc_per_site_st.shape[0])

        # Add an element to the legend for this species type
        leg_elements.append(Line2D([0], [0], marker='o', color=get_spec_type_col(spec_type), label=get_nice_lab(spec_type), markerfacecolor=get_spec_type_col(spec_type), markersize=10, lw=0))


    site_names = np.asarray(site_names)

    xs_agb = []
    ys_auc = []

    for site_ix, site_aucs in enumerate(auc_per_site.T):
        # For each site, plot scatters for AUCs of all species coloured by species type

        col_array = [get_spec_type_col(st) for st in spec_types]
        plt.scatter([site_sort_ix_agb[site_ix]] * len(site_aucs), site_aucs, c=col_array)

        # Save AGB and AUC for checking correlation later
        xs_agb.extend([site_agbs[site_sort_ix_agb[site_ix]]] * len(site_aucs))
        ys_auc.extend(site_aucs)

    xs_agb = np.asarray(xs_agb)
    ys_auc = np.asarray(ys_auc)

    # Check if there's any correlation between AGB and AUC
    rho, p = stats.pearsonr(xs_agb,ys_auc)
    print('Pearson corr between AGB and AUC: rho = {}, p = {}'.format(rho, p))

    plt.gca().legend(handles=leg_elements)
    plt.xticks(range(len(site_names)),site_names[site_sort_ix_agb], rotation=35)
    plt.xlabel('Site')
    plt.ylabel('AUC per species')
    plt.tight_layout()


if __name__ == '__main__':

    matplotlib.rc('font', size=18)

    plt.figure(figsize=(11,8))

    all_spec_types = ['herp','avi','avi-rl']

    plot_auc_by_site_fig(all_spec_types)

    plt.savefig(os.path.join('figs','fig_auc_by_site.pdf'),format='pdf')

    plt.show()
