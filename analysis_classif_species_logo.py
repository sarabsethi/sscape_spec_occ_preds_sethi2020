from pc_data_tools import load_pc_dataset, get_nparray_from_feats, save_classification_data, get_avi_specs_min_pres, get_avi_pcs_no_water_sites, get_herp_specs_min_pres, get_herp_pcs_no_water_sites, get_red_list_avi_specs_min_pres
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import os
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

TARGET_IX = os.getenv('PBS_ARRAY_INDEX')

hpc_arr_job = TARGET_IX is not None
if TARGET_IX is None:
    tqdm.write('No env variable found, processing all runs')

run = 0

all_pc_data_fs = ['pc_data_parsed_raw_audioset_feats_096s','pc_data_parsed_raw_audioset_feats_2s','pc_data_parsed_raw_audioset_feats_3s','pc_data_parsed_raw_audioset_feats_4s','pc_data_parsed_raw_audioset_feats_5s','pc_data_parsed_raw_audioset_feats_6s','pc_data_parsed_raw_audioset_feats_7s','pc_data_parsed_raw_audioset_feats_8s','pc_data_parsed_raw_audioset_feats_9s','pc_data_parsed_raw_audioset_feats_10s','pc_data_parsed_raw_audioset_feats_30s','pc_data_parsed_raw_audioset_feats_60s','pc_data_parsed_raw_audioset_feats_300s']
all_spec_types = ['avi-rl','avi','herp']


target_specs = None
#target_specs = ['short-tailed-babbler']
#all_pc_data_fs = all_pc_data_fs[::-1]

for spec_type in all_spec_types:
    for pc_data_fname in all_pc_data_fs:
        tqdm.write('Loading pc_data_fname {}'.format(pc_data_fname))
        audio_feat_name, all_sites, all_taxa, all_pcs = load_pc_dataset(pc_data_fname)

        if spec_type == 'herp':
            chosen_pcs = get_herp_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_herp_specs_min_pres(all_taxa, chosen_pcs)
        elif spec_type == 'avi':
            chosen_pcs = get_avi_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_avi_specs_min_pres(all_taxa, chosen_pcs)
        elif spec_type == 'avi-rl':
            chosen_pcs = get_avi_pcs_no_water_sites(all_pcs, all_sites)
            chosen_specs = get_red_list_avi_specs_min_pres(all_taxa, chosen_pcs)

        print([s.comm_name for s in chosen_specs])
        print('{} specs'.format(len(chosen_specs)))
        print('{} feats'.format(len(all_pc_data_fs)))

        all_pcs_sites = [pc.site.name for pc in chosen_pcs]
        _, all_pcs_groups = np.unique(all_pcs_sites, return_inverse=True)

        pcs_feats = np.asarray([pc.audio_feats for pc in chosen_pcs])

        for species in chosen_specs:
            if target_specs is not None and species.comm_name.lower().replace(' ','-') not in target_specs:
                continue

            pcs_spec_labs = np.asarray([0] * len(chosen_pcs))

            for ix, pc in enumerate(chosen_pcs):
                if spec_type == 'herp':
                    pcs_spec_labs[ix] = 1 if species in pc.herp_spec_comm else 0
                elif spec_type == 'avi' or spec_type == 'avi-rl':
                    pcs_spec_labs[ix] = 1 if species in pc.avi_spec_comm else 0

            cval = LeaveOneGroupOut()

            for k, (train_index, test_index) in enumerate(cval.split(pcs_feats, pcs_spec_labs, all_pcs_groups)):
                run = run + 1
                if hpc_arr_job and int(TARGET_IX) != run:
                    continue
                else:
                    tqdm.write('Running run {}: {}, {}, k={} '.format(run, pc_data_fname, species.comm_name, k))

                pc_feats_train, pc_feats_test = pcs_feats[train_index], pcs_feats[test_index]
                labs_train, labs_test = pcs_spec_labs[train_index], pcs_spec_labs[test_index]

                gmm_max_comps = 500
                gmm_max_comps = 1
                gmm_cov_type = 'diag'

                tqdm.write('{}: Fitting Bayesian GMMs with {} comps, {} covs'.format(species.comm_name, gmm_max_comps, gmm_cov_type))
                pres_feats_train = np.vstack(pc_feats_train[np.where((labs_train == 1))[0]])
                tqdm.write('{}: pres_feats_train {}'.format(species.comm_name, pres_feats_train.shape))
                pres_gmm_train = BayesianGaussianMixture(n_components=np.min([gmm_max_comps,pres_feats_train.shape[0]]),covariance_type=gmm_cov_type,random_state=10).fit(pres_feats_train)

                abs_feats_train = np.vstack(pc_feats_train[np.where((labs_train == 0))[0]])
                tqdm.write('{}: abs_feats_train {}'.format(species.comm_name, abs_feats_train.shape))
                abs_gmm_train = GaussianMixture(n_components=np.min([gmm_max_comps,abs_feats_train.shape[0]]),covariance_type=gmm_cov_type,random_state=10).fit(abs_feats_train)

                save_fname = '{}_bysn-gmm_{}_{}_{}_k{}'.format(species.comm_name.lower().replace(' ','-'), gmm_max_comps, gmm_cov_type, audio_feat_name, k)
                tqdm.write('Saving to file {}'.format(save_fname))
                save_classification_data(species, chosen_pcs, train_index, test_index, pcs_feats, pcs_spec_labs, pres_gmm_train, abs_gmm_train, save_fname, spec_type, save_dir='results_classifications_logo')
