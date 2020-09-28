import os
from pc_data_tools import load_classification_data, get_all_class_file_paths, load_spec_call_feats, get_llhood_ratio_score, get_nice_lab, get_spec_type_col
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib
import pickle
import matplotlib.patches as patches

matplotlib.rc('font', size=26)

show_all_specs = True

surf_type = 'median'
feat = 'raw_audioset_feats_3s'
score_type = 'p60'
results_dir = 'fig_data_logo'
all_spec_types = ['avi','herp','avi-rl']

panel_labels = ['(b)','(a)','(c)','(d)']
plot_specs = ['yellow-vented-bulbul','sooty-capped-babbler', 'tree-hole-frog']
plot_sort_ix = [1,0,2]

if not show_all_specs:
    fig, all_axes = plt.subplots(1,3,figsize=(36,7),sharey='all',sharex='all')
    all_axes = np.ravel(all_axes)
    plot_ix = 0

for spec_type in all_spec_types:

    f = 'classif_scores_sorted_{}_{}_{}.pickle'.format(feat,score_type,spec_type)
    load_path = os.path.join(results_dir, f)
    print(load_path)
    with open(load_path, 'rb') as f:
        auc_specs, auc_n_occs, auc_spec_agbs, aucs, auc_ks, train_scores_ks, train_labs_ks, test_scores_ks, test_lab_ks, test_hrs_ks, test_sites_ks = pickle.load(f)

    spec_sort_ix = np.argsort(np.argsort(aucs)[::-1])

    if show_all_specs:
        n_cols = 3
        n_rows = int(len(auc_specs)/n_cols) + 1
        fig, all_axes = plt.subplots(n_rows,n_cols,figsize=(36,7*n_rows),sharey='all',sharex='all')
        all_axes = np.ravel(all_axes)
        plot_ix = 0

    for spec_ix, spec in enumerate(auc_specs):
        if spec.comm_name.lower().replace(' ','-') not in plot_specs and not show_all_specs:
            continue

        if show_all_specs:
            plt.sca(all_axes[spec_sort_ix[plot_ix]])
        else:
            plt.sca(all_axes[plot_sort_ix[plot_ix]])
            plt.gca().text(-0.07, 1.12, panel_labels[plot_ix], transform=plt.gca().transAxes,fontsize=38, va='top', ha='right')

        plot_ix = plot_ix + 1

        spec_mean_auc = aucs[spec_ix]
        print('{}: spec_ix = {}'.format(spec.comm_name,spec_ix))

        flat_sites = np.asarray([item for sublist in test_sites_ks for item in sublist])
        flat_site_names = np.asarray([item.name for sublist in test_sites_ks for item in sublist])
        flat_site_abbrv_names = np.asarray([item.get_abbrv_name() for sublist in test_sites_ks for item in sublist])
        unq_site_names, unq_site_idxs = np.unique(flat_site_names, return_index=True)
        unq_sites = flat_sites[unq_site_idxs]
        unq_abbrv_names = flat_site_abbrv_names[unq_site_idxs]

        auc_k_site_names = []
        for site_list in test_sites_ks:
            auc_k_site_names.append([s.name for s in site_list])
        auc_k_site_names = np.asarray(auc_k_site_names)

        test_scores_ks = np.asarray(test_scores_ks)
        test_hrs_ks = np.asarray(test_hrs_ks)
        test_sites_ks = np.asarray(test_sites_ks)
        test_lab_ks = np.asarray(test_lab_ks)
        auc_ks = np.asarray(auc_ks)

        score_mat = []
        labs_mat = []
        all_hrs = np.asarray(range(24))
        for site_ix, site in enumerate(unq_sites):
            site_match_ix = np.where(auc_k_site_names == site.name)[1][spec_ix]

            t_scs = test_scores_ks[spec_ix,site_match_ix]
            t_hrs = test_hrs_ks[spec_ix,site_match_ix]
            t_labs = test_lab_ks[spec_ix,site_match_ix]

            hr_scores = []
            hr_labs = []
            for hr in all_hrs:
                hr_matches = np.where((t_hrs == hr))[0]
                hr_scs = t_scs[hr_matches]
                hr_lb = t_labs[hr_matches]

                if len(hr_scs) == 0:
                    hr_scores.append(np.nan)
                    hr_labs.append(np.nan)
                else:
                    hr_labs.append(np.mean(hr_lb))
                    if surf_type == 'mean':
                        hr_scores.append(np.mean(hr_scs))
                    elif surf_type == 'median':
                        hr_scores.append(np.median(hr_scs))
                    elif surf_type == 'max':
                        hr_scores.append(np.max(hr_scs))

            score_mat.append(np.asarray(hr_scores))
            labs_mat.append(np.asarray(hr_labs))

        score_mat = np.asarray(score_mat)
        labs_mat = np.asarray(labs_mat)

        site_agbs = [s.get_agb() for s in unq_sites]
        sort_ix_agb = np.argsort(site_agbs)
        score_mat_sorted = score_mat[sort_ix_agb]
        labs_mat_sorted = labs_mat[sort_ix_agb]

        bot_thresh = np.percentile(score_mat_sorted[~np.isnan(score_mat_sorted)],50)
        #top_thresh = np.percentile(score_mat_sorted[~np.isnan(score_mat_sorted)],95)

        current_cmap = matplotlib.cm.get_cmap('Blues_r')
        current_cmap.set_bad(color='gray')

        plt.matshow(score_mat_sorted, aspect='auto',vmin=bot_thresh,fignum=0,cmap=current_cmap)


        plt.colorbar(extend='min')

        for i, lab_row in enumerate(labs_mat_sorted):
            for j, l in enumerate(lab_row):
                if l > 0:
                    pt_c = 'r'
                    plt.scatter(j,i,marker='o',s=l*200,alpha=0.7,facecolors=pt_c,edgecolors=pt_c)
                    #rect = patches.Rectangle((j-0.5,i-0.5),1,1,linewidth=l*2,edgecolor='w',facecolor='none')
                    #plt.gca().add_patch(rect)

        plt.yticks(range(len(unq_site_names)),labels=unq_abbrv_names[sort_ix_agb])
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.xlabel('Hour of day')
        plt.ylabel('Site')
        plt.gca().set_title('{} ({} AUC)'.format(spec.comm_name,round(spec_mean_auc,2)), color=get_spec_type_col(spec_type))

        plt.gca().axhline(-0.5, c='k')
        plt.gca().axhline(1.5, c='k')
        plt.gca().axhline(3.5, c='k')
        plt.gca().axhline(8.5, c='k')
        plt.gca().axhline(10.5, c='k')



    if show_all_specs:
        for i in range(plot_ix, n_rows*n_cols):
            plt.sca(all_axes[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join('figs','{}_classif_surface_{}_{}_{}.pdf'.format(surf_type,feat,score_type,spec_type)),format='pdf')

if not show_all_specs:
    plt.tight_layout()
    plt.savefig(os.path.join('figs','classif_surface_site_time.pdf'),format='pdf')

plt.show()
