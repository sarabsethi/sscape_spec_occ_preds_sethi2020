import pickle
import os
import numpy as np
from pc_data_tools import get_all_class_file_paths, get_secs_per_audio_feat, get_nice_lab, get_spec_type_col
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
from sklearn.metrics import roc_auc_score


def do_auc_tscale_plot(lab_specs, lab_all_specs=False, all_spec_types=['herp','avi','avi-rl'], offset_aucs=False, score_type='p60', results_dir='fig_data_logo', print_all=False):
    '''
    Plot how AUC per species varies with timescale of audio features
    '''

    # Choose which timescales to plot
    f_exts = ['096','2','3','4','5','6','7','8','9','10','30','60','300']

    max_auc = 0
    for spec_ix, spec_type in enumerate(all_spec_types):

        # Option to plot increase in AUC over non-audio AGB based predictions
        if offset_aucs:
            no_audio_f = 'no_audio_classif_scores_{}.pickle'.format(spec_type)
            with open(os.path.join(results_dir, no_audio_f), 'rb') as f:
                _, _, _, baseline_aucs, _, _ = pickle.load(f)


        # Load classification results from stored files
        all_fnames = ['classif_scores_sorted_raw_audioset_feats_{}s_{}_{}.pickle'.format(f,score_type,spec_type) for f in f_exts]

        all_specs = None
        all_auc_ks = []
        all_test_scores_ks = []
        all_test_labs_ks = []
        for f in all_fnames:
            load_path = os.path.join(results_dir, f)
            with open(load_path, 'rb') as f:
                all_specs_, _, _, all_auc_, all_auc_ks_, _, _, all_test_scores_ks_, all_test_labs_ks_, _, _ = pickle.load(f)
            if all_specs is None: all_specs = all_specs_

            all_auc_ks.append(all_auc_ks_)
            all_test_scores_ks.append(all_test_scores_ks_)
            all_test_labs_ks.append(all_test_labs_ks_)

        all_auc_ks = np.asarray(all_auc_ks)
        all_test_scores_ks = np.asarray(all_test_scores_ks)
        all_test_labs_ks = np.asarray(all_test_labs_ks)
        print('all_auc_ks shape {}'.format(all_auc_ks.shape))
        print('all_test_scores_ks shape {}'.format(all_test_scores_ks.shape))
        print('all_test_labs_ks shape {}'.format(all_test_labs_ks.shape))

        # Extract seconds per feature used in each results file
        secs = []
        for f in all_fnames:
            f_s = f.split('_')[-3].split('s')[0]
            secs.append(get_secs_per_audio_feat('raw_audioset_feats_{}s'.format(f_s)))

        # Array to store which features were the best performing across all species
        max_secs = []

        # Loop through each species in turn
        for s_ix, spec in enumerate(all_specs):
            ys = []

            # Get AUCs for the given species
            spec_test_scores_ks = all_test_scores_ks[:,s_ix]
            spec_test_labs_ks = all_test_labs_ks[:,s_ix]
            spec_auc_ks = all_auc_ks[:,s_ix]

            # Get mean AUC across all K folds for the species (for each feature timescale)
            for x_ix, x in enumerate(secs):
                mean_auc = np.nanmean(spec_auc_ks[x_ix])
                ys.append(mean_auc)

            # If offset_aucs set, offset AUC by that attained by a model based on AGB values only
            if offset_aucs:
                ys = ys - baseline_aucs[s_ix]

            # Calculate null distributions for AUC values to assign p values for each real AUC result
            null_aucs = []
            null_perms = 100
            np.random.seed(42)
            for test_scores_k, test_labs_k in zip(spec_test_scores_ks.T, spec_test_labs_ks.T):
                if len(np.unique(test_labs_k[2])) > 1:
                    labs_shuffled = np.copy(test_labs_k[2])

                    k_aucs = []
                    for n in range(null_perms):
                        np.random.shuffle(labs_shuffled)
                        k_aucs.append(roc_auc_score(labs_shuffled,test_scores_k[2]))
                    null_aucs.append(k_aucs)
                else:
                    # If there's only one class (e.g., the species is never present or absent) then AUC is undefined
                    null_aucs.append([np.nan] * null_perms)

            # Calculate p value of true AUC based on a null distribution
            null_aucs = np.asarray(null_aucs)
            null_aucs = np.nanmean(null_aucs,axis=0)
            nulls_higher = null_aucs[null_aucs > ys[2]]
            auc_p_val = len(nulls_higher) / len(null_aucs)
            print('{} AUC = {} (p = {})'.format(spec.comm_name, np.round(ys[2],2), np.round(auc_p_val,2)))

            c = get_spec_type_col(spec_type)

            # Annotate the chosen species in lab_specs
            spec_str = spec.comm_name.lower().replace(' ','-')
            if lab_all_specs or spec_str in lab_specs:
                lw = 1
                if not lab_all_specs:
                    lw = 3
                alpha = 1
                plt.text(secs[-1]-10, ys[-1], spec.comm_name, horizontalalignment='right',verticalalignment='top', fontsize=14, color=c)

                if spec_str == 'sooty-capped-babbler' or spec_str == 'bold-striped-tit-babbler' or spec_str == 'tree-hole-frog' or spec_str == 'rhinoceros-hornbill':
                    print(ys)
            else:
                alpha = 0.5
                lw = 1

            if np.max(ys) > max_auc: max_auc = np.max(ys)

            max_ix = np.argmax(ys)
            max_secs.append(secs[max_ix])

            ls = '-'
            if auc_p_val > 0.05: ls = '--'

            # Plot a line showing AUC for the given species across different timescales of audio features
            p = plt.plot(secs,ys, lw=lw, ls=ls, alpha=alpha, c=c,label=get_nice_lab(spec_type) if s_ix == 0 else '')

        bins = secs + [(secs[-1]+1)]
        hist, _ = np.histogram(max_secs,bins)
        print(hist)

    print('Max AUC is {}'.format(max_auc))

    # Set x limits on the plot
    plt.xlim([secs[0],secs[-1]])
    plt.xscale('log')
    plt.gca().xaxis.grid(True,alpha=0.3)

    # Legend for the plot
    leg = plt.legend(loc='lower left')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2)
        legobj.set_alpha(1)

    # Labelling of x axis
    secs = np.asarray(secs)
    show_xtlabs = [0,1,2,4,6,9,10,11,12]
    xtls = []
    for s_ix, s in enumerate(secs):
        if s_ix in show_xtlabs:
            xtls.append(s)
        else:
            xtls.append('')
    xts = secs[show_xtlabs]
    plt.xticks(secs, labels=xtls)
    plt.xlabel('\nAudio feature timescale (s)')

    # Labelling of y axis
    if offset_aucs:
        plt.ylabel('AUC (gain over naive estimator, {})'.format(get_nice_lab(score_type)))
        plt.gca().axhline(0,lw=3,alpha=0.6,ls='--',c='k')
    else:
        plt.ylabel('AUC ({})'.format(get_nice_lab(score_type)))

    #plt.gca().axvline(2.88,alpha=0.2,lw=3,c='k')

if __name__ == '__main__':
    matplotlib.rc('font', size=18)

    fig = plt.figure(figsize=(15,9))

    lab_specs = np.asarray(['sooty-capped-babbler', 'tree-hole-frog','rhinoceros-hornbill','bold-striped-tit-babbler'])


    do_auc_tscale_plot(lab_specs, print_all=True)

    plt.savefig(os.path.join('figs','fic_compare_classif_feats.pdf'),format='pdf')

    plt.show()
