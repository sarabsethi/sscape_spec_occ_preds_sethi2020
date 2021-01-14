import os
import pickle
import numpy as np
from pc_data_tools import get_all_class_file_paths, load_classification_data, get_secs_per_audio_feat, get_llhood_ratio_score, load_pc_spectro
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib

'''
Plot how likelihood ratios (on a per feature resolution) change across a given recording,
overlaid on a spectrogram of the recording
'''


matplotlib.rc('font', size=20)

spec = 'yellow-bellied-prinia'
feat = 'raw_audioset_feats_3s'
score_type = 'p70'

all_fig_pc_fnames = ['PC0001_0315','PC0001_0736']
all_mins = [10,19]
all_fig_ks = [1,1]

n_plts = len(all_fig_pc_fnames)
fig, axes = plt.subplots(n_plts,1,figsize=(15,5*n_plts))
axes = np.ravel(axes)
plt_ix = 0

for pc_fname, chosen_k, chosen_min in zip(all_fig_pc_fnames, all_fig_ks, all_mins):
    # Find the classification results for the given point count
    result_f_paths = get_all_class_file_paths([feat,spec], k=chosen_k)
    print(result_f_paths)

    load_f_path = result_f_paths[0]
    species, avi_pcs_no_water, train_index, test_index, pcs_feats, pcs_spec_labs, pres_gmm_train, abs_gmm_train = load_classification_data(load_f_path)

    test_feats = pcs_feats[test_index]
    test_pcs = np.asarray(avi_pcs_no_water)[test_index]

    labs_test = pcs_spec_labs[test_index]

    # Find the chosen point count in the test set
    test_pc_fnames = np.asarray([pc.audio_fname for pc in test_pcs])

    plt.sca(axes[plt_ix])
    print(pc_fname)
    pc_test_ix = np.where((test_pc_fnames == pc_fname.upper()))[0][0]

    true_lab = labs_test[pc_test_ix]
    pc = test_pcs[pc_test_ix]

    # Compute per audio feature likelihood ratios
    test_f = test_feats[pc_test_ix]
    samps = pres_gmm_train.score_samples(test_f) - abs_gmm_train.score_samples(test_f)

    # Get summary score using chosen score_type
    summ_score = get_llhood_ratio_score(test_f, pres_gmm_train, abs_gmm_train, score_type)

    xs = np.asarray(list(range(len(samps))))
    spf = get_secs_per_audio_feat(feat)
    xs_mins = np.asarray(xs*spf/60)

    plt_ixs = np.where((np.logical_and(xs_mins > chosen_min,xs_mins < chosen_min+1)))[0]
    plt_ixs = range(len(xs_mins))

    # Load pre-computed spectrogram for the given point count and plot it
    spectro = load_pc_spectro(pc.audio_fname)[0]
    spectro = spectro[plt_ixs,:]
    plt.matshow(spectro.T, aspect='auto', origin='lower',fignum=0)
    plt.xticks([])

    plt.gca().twinx()
    plt.gca().twiny()

    plt.gca().axhline(summ_score,alpha=0.5)

    # Plot per feature likelihood ratios in red on top of spectrogram
    c='r'
    ls = '-'
    if true_lab == 0: ls = '--'
    plt.plot(xs_mins[plt_ixs],samps[plt_ixs],c=c,ls=ls)
    plt.xlim([np.min(xs_mins[plt_ixs]),np.max(xs_mins[plt_ixs])])

    plt.ylabel(r'$C_{ij}$')
    plt.gca().xaxis.set_ticks_position('bottom')

    plt.title('{} ({})'.format(pc.audio_fname, pc.id))

    plt_ix = plt_ix + 1

plt.tight_layout()
plt.show()
