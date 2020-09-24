import os
from pc_data_tools import load_classification_data, get_all_class_file_paths, load_spec_call_feats, get_llhood_ratio_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pickle


TARGET_IX = os.getenv('PBS_ARRAY_INDEX')
hpc_arr_job = TARGET_IX is not None
if TARGET_IX is None:
    print('No env variable found, processing all runs')
run = 0

results_dir = 'results_classifications_logo'
out_dir = 'fig_data_logo'

all_score_types = ['min', 'p10','p20','p30','p40','p50', 'p60', 'p70', 'p80', 'p90', 'max', 'mean']
all_feats = ['raw_audioset_feats_096s','raw_audioset_feats_2s','raw_audioset_feats_3s','raw_audioset_feats_4s','raw_audioset_feats_5s','raw_audioset_feats_6s','raw_audioset_feats_7s','raw_audioset_feats_8s','raw_audioset_feats_9s','raw_audioset_feats_10s','raw_audioset_feats_30s','raw_audioset_feats_60s','raw_audioset_feats_300s']
all_spec_types = ['avi-rl','avi','herp']

for spec_type in all_spec_types:
    for score_type in all_score_types:
        for feat in all_feats:
            run = run + 1
            if hpc_arr_job and int(TARGET_IX) != run:
                continue
            else:
                print('Running run {}: {} {}'.format(run, feat, score_type))

            result_f_paths = get_all_class_file_paths(string_filters=feat, k=-1, spec_type=spec_type, classif_res_dir=results_dir)
            result_f_path_stems = np.unique(np.asarray(['_'.join(path.split('_')[:-1]) for path in result_f_paths]))
            print(result_f_path_stems)

            all_auc = np.asarray([])
            all_auc_ks = []
            all_specs = np.asarray([])
            all_spec_agbs = np.asarray([])
            all_n_occs = np.asarray([])
            all_train_scores_ks = []
            all_train_lab_ks = []
            all_test_scores_ks = []
            all_test_lab_ks = []
            all_test_hrs_ks = []

            tot_k = 11

            all_k_sites = []
            for load_f_path_stem in tqdm(result_f_path_stems):

                auc_ks = []
                spec_k_sites = []
                train_score_ks = []
                train_lab_ks = []
                test_score_ks = []
                test_lab_ks = []
                test_hr_ks = []
                for k in range(tot_k):
                    load_f_path = load_f_path_stem + '_k{}.pickle'.format(k)
                    if not os.path.exists(load_f_path):
                        print('ERROR NO FILE FOUND: {}'.format(load_f_path))
                        raise Exception('NO FILE')

                    species, chosen_pcs, train_index, test_index, pcs_feats, pcs_spec_labs, pres_gmm_train, abs_gmm_train = load_classification_data(load_f_path)

                    labs_train = pcs_spec_labs[train_index]
                    labs_test = pcs_spec_labs[test_index]

                    site_name = chosen_pcs[test_index[0]].site.name

                    spec_k_sites.append(chosen_pcs[test_index[0]].site)

                    scores_train = np.asarray([get_llhood_ratio_score(feat_coll, pres_gmm_train, abs_gmm_train, score_type) for feat_coll in pcs_feats[train_index]])
                    scores_test = np.asarray([get_llhood_ratio_score(feat_coll, pres_gmm_train, abs_gmm_train, score_type) for feat_coll in pcs_feats[test_index]])
                    scores_train = scores_train.reshape(-1, 1)
                    scores_test = scores_test.reshape(-1, 1)

                    hrs_test = [pc.dt.hour for pc in np.asarray(chosen_pcs)[test_index]]

                    train_score_ks.append(scores_train)
                    train_lab_ks.append(labs_train)
                    test_score_ks.append(scores_test)
                    test_lab_ks.append(labs_test)
                    test_hr_ks.append(hrs_test)

                    if len(np.unique(labs_test)) == 1:
                        print('k = {}, only one class in test set (label = {}, test site {})'.format(k,np.unique(labs_test),site_name))
                        auc_ks.append(np.nan)
                    else:
                        auc_k_ = roc_auc_score(labs_test, scores_test)
                        tqdm.write('k = {}, auc_k_ = {}'.format(k, auc_k_))
                        auc_ks.append(auc_k_)

                auc = np.nanmean(auc_ks)

                unq_site_names, site_indices = np.unique([p.site.name for p in chosen_pcs],return_index=True)
                unq_sites = np.asarray([p.site for p in chosen_pcs])[site_indices]
                unq_site_agbs = [s.get_agb() for s in unq_sites]

                if spec_type == 'herp':
                    pres_site_names = np.asarray([p.site.name for p in chosen_pcs if species in p.herp_spec_comm])
                elif spec_type == 'avi' or spec_type == 'avi-rl':
                    pres_site_names = np.asarray([p.site.name for p in chosen_pcs if species in p.avi_spec_comm])

                tot_hist = [np.sum(np.asarray([p.site.name for p in chosen_pcs]) == s) for s in unq_site_names]
                spec_hist = [np.sum(pres_site_names == s) for s in unq_site_names]
                spec_hist_weighted = np.asarray(spec_hist) / np.asarray(tot_hist)
                spec_agb_weighted = np.sum(spec_hist_weighted * unq_site_agbs)

                tqdm.write('{}, auc = {}'.format(species.comm_name, round(auc,2)))

                all_auc = np.hstack((all_auc,auc))
                all_auc_ks.append(auc_ks)
                all_train_lab_ks.append(train_lab_ks)
                all_test_lab_ks.append(test_lab_ks)
                all_test_hrs_ks.append(test_hr_ks)
                all_train_scores_ks.append(train_score_ks)
                all_test_scores_ks.append(test_score_ks)
                all_specs = np.hstack((all_specs,species))
                all_spec_agbs = np.hstack((all_spec_agbs,spec_agb_weighted))
                all_n_occs = np.hstack((all_n_occs,np.sum(pcs_spec_labs)))
                all_k_sites.append(spec_k_sites)

            fig_savef = 'classif_scores_sorted_{}_{}_{}'.format(feat,score_type,spec_type)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            save_path = os.path.join(out_dir,'{}.pickle'.format(fig_savef))
            with open(save_path, 'wb') as f:
                pickle.dump([all_specs, all_n_occs, all_spec_agbs, all_auc, all_auc_ks, all_train_scores_ks, all_train_lab_ks, all_test_scores_ks, all_test_lab_ks, all_test_hrs_ks, all_k_sites], f)
