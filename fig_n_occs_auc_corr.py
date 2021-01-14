import pickle
import os
import numpy as np
from pc_data_tools import load_pc_dataset, get_secs_per_audio_feat, get_avi_pcs_no_water_sites,get_avi_specs_min_pres,get_herp_pcs_no_water_sites,get_herp_specs_min_pres, get_nice_lab, get_red_list_avi_specs_min_pres, get_spec_type_col
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib


def plot_noccs_fig(lab_specs, lab_all_specs, all_spec_types, score_type='p60', feat='raw_audioset_feats_3s', results_dir='fig_data_logo'):
    '''
    Plot the relationship between AUC of species occurrence predictions and number of occurrences of each species
    '''

    xs_auc = []
    xs_specs = []
    all_n_occs = []
    xs_spec_types = []

    for spec_type in all_spec_types:
        # Load classification results
        f = 'classif_scores_sorted_{}_{}_{}.pickle'.format(feat,score_type,spec_type)
        load_path = os.path.join(results_dir, f)
        with open(load_path, 'rb') as f:
            auc_specs, spec_n_occs, _, aucs, auc_ks, _, _, _, _, _, _ = pickle.load(f)

        auc_spec_names = np.asarray([s.comm_name for s in auc_specs])

        audio_feat_name, all_sites, all_taxa, all_pcs = load_pc_dataset('pc_data_parsed_{}'.format(feat))

        # Load the appropriate point counts and species lists
        if spec_type == 'avi':
            chosen_pcs = get_avi_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_avi_specs_min_pres(all_taxa, chosen_pcs)
        elif spec_type == 'avi-rl':
            chosen_pcs = get_avi_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_red_list_avi_specs_min_pres(all_taxa, chosen_pcs)
        elif spec_type == 'herp':
            chosen_pcs = get_herp_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_herp_specs_min_pres(all_taxa, chosen_pcs)

        # Loop through each species extracting number of occurrences and species AUCs
        for spec_ix, spec in enumerate(chosen_specs):
            auc_ix = np.where((auc_spec_names == spec.comm_name))[0]
            auc = aucs[auc_ix][0]

            xs_auc.append(auc)
            xs_specs.append(spec)
            xs_spec_types.append(spec_type)
            all_n_occs.append(spec_n_occs[spec_ix])

    xs_auc = np.asarray(xs_auc)

    # Calculate correlation between AUC and number of occurrences per species
    plt_ys = all_n_occs
    plt_rho, plt_p = stats.pearsonr(xs_auc,plt_ys)

    for s_ix, s in enumerate(xs_specs):
        c = get_spec_type_col(xs_spec_types[s_ix])

        pt_sz = 50
        pt_a = 0.3
        spec_str = s.comm_name.lower().replace(' ','-')

        # Label species on the scatter plot
        if lab_all_specs or spec_str in lab_specs:
            pt_sz = 50
            if not lab_all_specs:
                pt_sz = 90

            # Hack to make sure species annotations don't overlap
            ha = 'left'
            hoffs = 0.008
            voffs=0
            if xs_auc[s_ix] > 0.82 or 'rough-guardian-frog' in spec_str:
                ha = 'right'
                hoffs = -hoffs

            plt.text(xs_auc[s_ix]+hoffs,plt_ys[s_ix]+voffs,s.comm_name,color=c,verticalalignment='center',horizontalalignment=ha)
            pt_a = 1

        # Scatter point for given species
        plt.scatter(xs_auc[s_ix],plt_ys[s_ix],c=c,s=pt_sz,alpha=pt_a)

    # Plot line of best fit
    m, c = np.polyfit(xs_auc, plt_ys, 1)
    plt.plot(xs_auc, m*xs_auc + c, c='k', alpha=0.2)

    plt.ylabel('Number of occurrences in point counts')
    plt.xlabel('\nAUC ({}s per feature, {})'.format(get_secs_per_audio_feat(feat),get_nice_lab(score_type)))

    if plt_p < 0.001:
        p_txt = 'p < 0.001'
    else:
        p_txt = 'p = {}'.format(round(plt_p,6))

    plt.text(sorted(xs_auc)[0], sorted(plt_ys)[-1], 'Pearson correlation:\n' + r'$\rho$ = {}, {}'.format(round(plt_rho,2), p_txt), verticalalignment='top')

    plt.tight_layout()


if __name__ == '__main__':

    matplotlib.rc('font', size=18)

    plt.figure(figsize=(11,8))

    lab_specs = np.asarray(['asian-red-eyed-bulbul','sooty-capped-babbler', 'tree-hole-frog','rough-guardian-frog','rhinoceros-hornbill'])

    lab_all_specs = False
    all_spec_types = ['herp','avi','avi-rl']

    plot_noccs_fig(lab_specs, lab_all_specs, all_spec_types)

    plt.savefig(os.path.join('figs','fig_n_occs_auc_corr.pdf'),format='pdf')

    plt.show()
