import pickle
import os
import numpy as np
from pc_data_tools import load_pc_dataset, get_secs_per_audio_feat, get_avi_pcs_no_water_sites, get_avi_specs_min_pres, get_herp_pcs_no_water_sites, get_herp_specs_min_pres, get_nice_lab, get_red_list_avi_specs_min_pres, get_spec_type_col
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib


def plot_chi2_fig(lab_specs, lab_all_specs, all_spec_types, score_type='p60', feat='raw_audioset_feats_3s', results_dir='fig_data_logo'):
    '''
    Plot figure showing correlation between AUC of species prediction and chi^2 statistics
    '''

    xs_auc = []
    ys_site_chi2 = []
    ys_hr_chi2 = []
    ys_site_hr_chi2 = []
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

        # Load point count dataset
        audio_feat_name, all_sites, all_taxa, all_pcs = load_pc_dataset('pc_data_parsed_{}'.format(feat))

        if spec_type == 'avi':
            chosen_pcs = get_avi_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_avi_specs_min_pres(all_taxa, chosen_pcs)
        elif spec_type == 'avi-rl':
            chosen_pcs = get_avi_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_red_list_avi_specs_min_pres(all_taxa, chosen_pcs)
        elif spec_type == 'herp':
            chosen_pcs = get_herp_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_herp_specs_min_pres(all_taxa, chosen_pcs)

        for spec_ix, spec in enumerate(chosen_specs):
            # Compute chi2 statistics for each species in turn

            # Create a vector with labels for each point count - 1 if species is present, 0 if absent
            pcs_spec_labs = np.asarray([0] * len(chosen_pcs))
            for ix, pc in enumerate(chosen_pcs):
                if spec_type == 'avi' or spec_type == 'avi-rl':
                    pcs_spec_labs[ix] = 1 if spec in pc.avi_spec_comm else 0
                elif spec_type == 'herp':
                    pcs_spec_labs[ix] = 1 if spec in pc.herp_spec_comm else 0

            pres_ixs = np.where((pcs_spec_labs == 1))[0]
            abs_ixs = np.where((pcs_spec_labs == 0))[0]

            # Get mean AUC for species
            auc_ix = np.where((auc_spec_names == spec.comm_name))[0]
            auc = aucs[auc_ix][0]
            #print('{} AUC {}'.format(spec.comm_name,auc))

            # Create vectors for point count sites, hours, and site hours
            pc_sites = np.asarray([pc.site.name for pc in chosen_pcs])
            pc_hrs = np.asarray([pc.dt.hour for pc in chosen_pcs])
            pc_site_hrs = np.asarray(['{} {}'.format(pc.site.name,pc.dt.hour) for pc in chosen_pcs])

            # Create contingency table based on point count sites
            unq_sites = np.unique(pc_sites)
            site_cont_tab = []
            for site in unq_sites:
                site_ixs = np.where((pc_sites == site))[0]
                pres_s_ixs = np.intersect1d(site_ixs,pres_ixs)
                abs_s_ixs = np.intersect1d(site_ixs,abs_ixs)
                site_cont_tab.append([len(pres_s_ixs), len(abs_s_ixs)])
            site_cont_tab = np.asarray(site_cont_tab)

            # Calculate chi^2 statistic on contingency table
            s_chi2, s_p, _, _ = stats.chi2_contingency(site_cont_tab)

            # Create contingency table based on point count hour of days
            unq_hrs = np.unique(pc_hrs)
            hr_cont_tab = []
            for hr in unq_hrs:
                hr_ixs = np.where((pc_hrs == hr))[0]
                pres_hr_ixs = np.intersect1d(hr_ixs,pres_ixs)
                abs_hr_ixs = np.intersect1d(hr_ixs,abs_ixs)
                hr_cont_tab.append([len(pres_hr_ixs), len(abs_hr_ixs)])
            hr_cont_tab = np.asarray(hr_cont_tab)

            # Calculate chi^2 statistic on contingency table
            h_chi2, h_p, _, _ = stats.chi2_contingency(hr_cont_tab)

            # Create contingency table based on point count site/hour combos
            unq_site_hrs = np.unique(pc_site_hrs)
            site_hr_cont_tab = []
            for site_hr in unq_site_hrs:
                site_hr_ixs = np.where((pc_site_hrs == site_hr))[0]
                pres_site_hr_ixs = np.intersect1d(site_hr_ixs,pres_ixs)
                abs_site_hr_ixs = np.intersect1d(site_hr_ixs,abs_ixs)
                site_hr_cont_tab.append([len(pres_site_hr_ixs), len(abs_site_hr_ixs)])
            site_hr_cont_tab = np.asarray(site_hr_cont_tab)

            # Calculate chi^2 statistic on contingency table
            s_h_chi2, s_h_p, _, _ = stats.chi2_contingency(site_hr_cont_tab)

            xs_auc.append(auc)
            ys_site_chi2.append(s_chi2)
            ys_hr_chi2.append(h_chi2)
            ys_site_hr_chi2.append(s_h_chi2)
            xs_specs.append(spec)
            xs_spec_types.append(spec_type)
            all_n_occs.append(spec_n_occs[spec_ix])

    xs_auc = np.asarray(xs_auc)
    ys_hr_chi2 = np.asarray(ys_hr_chi2)
    ys_site_chi2 = np.asarray(ys_site_chi2)
    ys_site_hr_chi2 = np.asarray(ys_site_hr_chi2)

    print('AUC max: {}, min {}'.format(np.max(xs_auc), np.min(xs_auc)))

    # Calculate pearson correlations between chi^2 statistics and AUCs
    s_rho, s_p = stats.pearsonr(xs_auc,ys_site_chi2)
    print('site {} {}'.format(s_rho,s_p))

    h_rho, h_p = stats.pearsonr(xs_auc,ys_hr_chi2)
    print('hour {} {}'.format(h_rho,h_p))

    sh_rho, sh_p = stats.pearsonr(xs_auc,ys_site_hr_chi2)
    print('site hour {} {}'.format(sh_rho,sh_p))

    noccs_rho, noccs_p = stats.pearsonr(xs_auc,all_n_occs)
    print('all_n_occs {} {}'.format(noccs_rho,noccs_p))

    plt_ys = ys_hr_chi2
    plt_rho, plt_p = stats.pearsonr(xs_auc,plt_ys)

    avi_chi2s = []
    avi_rl_chi2s = []
    herp_chi2s = []
    avi_aucs = []
    avi_rl_aucs = []
    herp_aucs = []
    for s_ix, s in enumerate(xs_specs):
        # Plot points on scatter for each species in turn

        c = get_spec_type_col(xs_spec_types[s_ix])

        pt_sz = 50
        pt_a = 0.3
        spec_str = s.comm_name.lower().replace(' ','-')

        # Annotate chosen species on scatter plot
        if lab_all_specs or spec_str in lab_specs:
            pt_sz = 50
            if not lab_all_specs:
                pt_sz = 90

            # Hack to make species name annotations non-overlapping
            ha = 'left'
            hoffs = 0.008
            voffs=0
            if xs_auc[s_ix] > 0.82 or 'rough-guardian-frog' in spec_str:
                ha = 'right'
                hoffs = -hoffs

            plt.text(xs_auc[s_ix]+hoffs,plt_ys[s_ix]+voffs,s.comm_name,color=c,verticalalignment='center',horizontalalignment=ha)
            pt_a = 1

        plt.scatter(xs_auc[s_ix],plt_ys[s_ix],c=c,s=pt_sz,alpha=pt_a)

        if xs_spec_types[s_ix] == 'avi':
            avi_chi2s.append(plt_ys[s_ix])
            avi_aucs.append(xs_auc[s_ix])
        elif xs_spec_types[s_ix] == 'herp':
            herp_chi2s.append(plt_ys[s_ix])
            herp_aucs.append(xs_auc[s_ix])
        elif xs_spec_types[s_ix] == 'avi-rl':
            avi_rl_chi2s.append(plt_ys[s_ix])
            avi_rl_aucs.append(xs_auc[s_ix])

    # Perform T test between species types to determine if one type is more temporally niched / spatially niched than the other
    t_stat_chi2, t_p_chi2 = stats.ttest_ind(avi_chi2s,herp_chi2s)
    t_stat_auc, t_p_auc = stats.ttest_ind(avi_aucs,herp_aucs)
    t_stat_auc_rl, t_p_auc_rl = stats.ttest_ind(avi_aucs,avi_rl_aucs)
    print('Herp vs avi chi2 t-test: stat = {}, p = {}. Mean avi: {} herp: {}'.format(t_stat_chi2, t_p_chi2, np.mean(avi_chi2s),np.mean(herp_chi2s)))
    print('Herp vs avi auc t-test: stat = {}, p = {}. Mean avi: {} herp: {}'.format(t_stat_auc, t_p_auc, np.mean(avi_aucs),np.mean(herp_aucs)))
    print('Avi vs avi-rl auc t-test: stat = {}, p = {}. Mean avi: {} avi-rl: {}'.format(t_stat_auc_rl, t_p_auc_rl, np.mean(avi_aucs),np.mean(avi_rl_aucs)))

    # Plot line of best fit
    m, c = np.polyfit(xs_auc, plt_ys, 1)
    plt.plot(xs_auc, m*xs_auc + c, c='k', alpha=0.2)

    plt.ylabel('Occurrence/hour $\chi^2$ statistic')
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

    plot_chi2_fig(lab_specs, lab_all_specs, all_spec_types)

    plt.savefig(os.path.join('figs','fic_chi2_auc.pdf'),format='pdf')

    plt.show()
