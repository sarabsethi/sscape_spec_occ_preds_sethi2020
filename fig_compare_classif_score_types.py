import os
import pickle
import numpy as np
from pc_data_tools import get_secs_per_audio_feat, get_nice_lab
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib

def plot_feat_score_type_comp(offset_aucs=False, results_dir='fig_data_logo', all_spec_types=['avi','herp','avi-rl']):
    '''
    Compare different methods of aggregating likelihood ratios (at the per feature level) into one
    classification confidence per audio file
    '''

    all_feats = ['raw_audioset_feats_096s','raw_audioset_feats_2s','raw_audioset_feats_3s','raw_audioset_feats_4s','raw_audioset_feats_5s','raw_audioset_feats_6s','raw_audioset_feats_7s','raw_audioset_feats_8s','raw_audioset_feats_9s','raw_audioset_feats_10s','raw_audioset_feats_30s','raw_audioset_feats_60s','raw_audioset_feats_300s']
    all_score_types = ['min', 'p10', 'p20','p30', 'p40', 'p50', 'p60','p70', 'p80', 'p90','max', 'mean']

    all_hists = []
    all_spec_aucs = []
    all_spec_mean_aucs = []
    all_specs = []

    for feat in all_feats:
        # Loop through each feature time resolution
        st_aucs = []
        for score_type in all_score_types:
            # Loop through different methods for obtaining classification scores
            all_spec_aucs = []

            for spec_type in all_spec_types:
                # Loop through species types

                # Option to offset AUCs by that possible without audio data (just using AGB data of each site)
                if offset_aucs:
                    no_audio_f = 'no_audio_classif_scores_{}.pickle'.format(spec_type)
                    with open(os.path.join(results_dir, no_audio_f), 'rb') as f:
                        _, _, _, baseline_aucs, _, _ = pickle.load(f)

                # Load classification results
                fname = 'classif_scores_sorted_{}_{}_{}.pickle'.format(feat, score_type, spec_type)
                load_path = os.path.join(results_dir, fname)
                with open(load_path, 'rb') as f:
                    all_specs, _, _, all_auc_, all_auc_ks_, _, _, _, _, _, _ = pickle.load(f)

                if offset_aucs:
                    all_auc_ = all_auc_ - baseline_aucs

                all_spec_aucs.extend(all_auc_)

            # Calculate mean AUC for given score type
            spec_auc_means = np.nanmean(all_spec_aucs, axis=0)
            st_aucs.append(spec_auc_means)

        all_spec_mean_aucs.append(st_aucs)

    # Transpose matrix of mean AUCs per score type
    all_spec_mean_aucs = np.asarray(all_spec_mean_aucs).T

    # Place an asterix over the score type / feature combo that gives the max mean AUC across all species
    max_auc_ix = np.unravel_index(np.argmax(all_spec_mean_aucs),all_spec_mean_aucs.shape)
    plt.scatter(max_auc_ix[1], max_auc_ix[0], marker='*',s=250,edgecolors='k',facecolors='none')

    # Plot matrix and label score types and feature timescales
    plt.matshow(all_spec_mean_aucs, aspect='equal', fignum=0)
    plt.yticks(range(len(all_score_types)), labels=[get_nice_lab(l) for l in all_score_types])

    secs = [get_secs_per_audio_feat(f) for f in all_feats]
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xticks(range(len(secs)), labels=secs, fontsize=20,rotation=70)
    cb = plt.colorbar(ticklocation='left')
    if offset_aucs:
        cb.set_label('\nMean AUC gain (across all species)')
    else:
        cb.set_label('\nMean AUC (across all species)')

    plt.xlabel('Audio feature timescale (s)')
    plt.ylabel('Classification score method')

    plt.tight_layout()

    best_feat = all_feats[max_auc_ix[1]]
    best_score_type = all_score_types[max_auc_ix[0]]

    print('best_feat {}, best_score_type {} (mean AUC = {})'.format(best_feat,best_score_type,all_spec_mean_aucs[max_auc_ix]))

    return best_feat, best_score_type

if __name__ == '__main__':
    matplotlib.rc('font', size=28)
    fig = plt.figure(figsize=(13,10))

    plot_feat_score_type_comp()

    if not os.path.exists('figs'): os.makedirs('figs')
    plt.savefig(os.path.join('figs','fig_compare_classif_scores.pdf'),format='pdf')

    plt.show()
