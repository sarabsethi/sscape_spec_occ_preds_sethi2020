import os
from pc_data_tools import load_classification_data, get_all_class_file_paths
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pickle

'''
Predict species occurrence based only on site AGB (without using any soundscape data)
'''

all_spec_types = ['avi-rl','herp','avi']
results_dir = 'results_classifications_logo'

out_dir = 'fig_data_logo'

all_acd_dists = ['100m', '250m', '500m']

for acd_dist in all_acd_dists:
    for spec_type in all_spec_types:
        # This is just to get point count data from saved files - we don't actually use the audio data
        feat = 'raw_audioset_feats_096s'

        result_f_paths = get_all_class_file_paths(string_filters=feat, k=-1, spec_type=spec_type, classif_res_dir=results_dir)
        result_f_path_stems = np.unique(np.asarray(['_'.join(path.split('_')[:-1]) for path in result_f_paths]))
        print(result_f_path_stems)

        all_auc = []
        all_auc_ks = []
        all_specs = []
        all_spec_agbs = []
        all_n_occs = []

        tot_k = 11

        for load_f_path_stem in tqdm(result_f_path_stems):
            auc_ks = []
            for k in range(tot_k):
                # Loop through each fold of the cross validation classification task (uses the same splits as the soundscape predictions)
                load_f_path = load_f_path_stem + '_k{}.pickle'.format(k)
                if not os.path.exists(load_f_path):
                    print('ERROR NO FILE FOUND: {}'.format(load_f_path))
                    continue

                species, chosen_pcs, train_index, test_index, pcs_feats, pcs_spec_labs, pres_gmm_train, abs_gmm_train = load_classification_data(load_f_path)

                labs_train = pcs_spec_labs[train_index]
                labs_test = pcs_spec_labs[test_index]

                chosen_pcs = np.asarray(chosen_pcs)

                # Get AGB for each site
                all_site_names = np.asarray([pc.site.name for pc in chosen_pcs])
                all_site_agbs = np.asarray([pc.site.get_agb(average_size=acd_dist) for pc in chosen_pcs])
                unq_sites, unq_site_ixs = np.unique(all_site_names, return_index=True)
                unq_site_agbs = all_site_agbs[unq_site_ixs]

                # For each site determine which of the other sites has the most similar AGB
                nearest_sites = []
                for site_ix, site in enumerate(unq_sites):
                    site_agb = unq_site_agbs[site_ix]
                    sort_ix = np.argsort(np.abs(unq_site_agbs - site_agb))
                    ordered_nearest_sites = unq_sites[sort_ix]
                    nearest_sites.append(ordered_nearest_sites[ordered_nearest_sites!=site][0])
                nearest_sites = np.asarray(nearest_sites)


                all_pc_site_hrs = np.asarray(['{} {}'.format(pc.site.name,pc.dt.hour) for pc in chosen_pcs])
                all_unq_pc_site_hrs = np.unique(all_pc_site_hrs)

                train_pc_site_hrs = np.asarray(['{} {}'.format(pc.site.name,pc.dt.hour) for pc in chosen_pcs[train_index]])
                test_pc_site_hrs = np.asarray(['{} {}'.format(pc.site.name,pc.dt.hour) for pc in chosen_pcs[test_index]])

                # Go through each site/hour combo and make a prediction based on occurrences in the training dataset
                naive_sh_guesses = []
                for sh in all_unq_pc_site_hrs:
                    s = ' '.join(sh.split(' ')[:-1])
                    h = ''.join(sh.split(' ')[-1])
                    nearest_s = nearest_sites[unq_sites == s][0]
                    train_sh_ixs = np.where((train_pc_site_hrs == '{} {}'.format(nearest_s,h)))[0]

                    if len(train_sh_ixs) > 0:
                        train_sh_labs = labs_train[train_sh_ixs]
                        # Prediction is the mean of true occurrences in the training datast
                        # at the most similar site (in terms of AGB) at the same hour of day
                        naive_sh_guesses.append(np.mean(train_sh_labs))
                    else:
                        naive_sh_guesses.append(0)
                naive_sh_guesses = np.asarray(naive_sh_guesses)

                # Map predictions to data in the test set
                test_scores = []
                for sh in test_pc_site_hrs:
                    match_ix = np.where((all_unq_pc_site_hrs == sh))[0]
                    test_scores.append(naive_sh_guesses[match_ix])

                # If species is always present/absent AUC is undefined
                if len(np.unique(labs_test)) == 1:
                    print('k = {} only one unique test label {}, AUC = nan'.format(k, np.unique(labs_test)))
                    auc_ks.append(np.nan)
                    continue

                # Otherwise, calculate AUC metric
                auc_k_ = roc_auc_score(labs_test, test_scores)
                tqdm.write('k = {}, auc_k_ = {}'.format(k, auc_k_))
                auc_ks.append(auc_k_)

            auc = np.nanmean(auc_ks)

            # Determine AGB per site
            unq_site_names, site_indices = np.unique([p.site.name for p in chosen_pcs],return_index=True)
            unq_sites = np.asarray([p.site for p in chosen_pcs])[site_indices]
            unq_site_agbs = [s.get_agb(average_size=acd_dist) for s in unq_sites]

            # Determine sites at which the species was present
            if spec_type == 'herp':
                pres_site_names = np.asarray([p.site.name for p in chosen_pcs if species in p.herp_spec_comm])
            elif spec_type == 'avi' or spec_type == 'avi-rl':
                pres_site_names = np.asarray([p.site.name for p in chosen_pcs if species in p.avi_spec_comm])

            # Calculate histogram of how often species was found at each site, and from that the mean AGB that the species was encountered at
            tot_hist = [np.sum(np.asarray([p.site.name for p in chosen_pcs]) == s) for s in unq_site_names]
            spec_hist = [np.sum(pres_site_names == s) for s in unq_site_names]
            spec_hist_weighted = np.asarray(spec_hist) / np.asarray(tot_hist)
            spec_agb_weighted = np.sum(spec_hist_weighted * unq_site_agbs)

            tqdm.write('{}, auc = {}'.format(species.comm_name, round(auc,2)))

            all_auc = np.hstack((all_auc,auc))
            all_auc_ks.append(auc_ks)
            all_specs = np.hstack((all_specs,species))
            all_spec_agbs = np.hstack((all_spec_agbs,spec_agb_weighted))
            all_n_occs = np.hstack((all_n_occs,np.sum(pcs_spec_labs)))

        sort_idx = np.argsort(np.asarray(all_auc))

        # Save data to file
        savef = 'no_audio_classif_scores_{}_acd-{}'.format(spec_type, acd_dist)
        save_path = os.path.join(out_dir,'{}.pickle'.format(savef))
        with open(save_path, 'wb') as f:
            pickle.dump([all_specs, all_n_occs, all_spec_agbs, all_auc, all_auc_ks, sort_idx], f)
