import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
import copy
import scipy.stats
from pc_data_tools import get_spec_type_col

def plot_no_audio_comp(results_dir='fig_data_logo', all_spec_types=['avi','herp','avi-rl'], feat='raw_audioset_feats_3s', score_type='p60', lab_specs=['asian-red-eyed-bulbul','sooty-capped-babbler', 'tree-hole-frog','rough-guardian-frog','rhinoceros-hornbill']):
    
    lab_specs = []
    all_spec_names = None
    all_aucs_na = []
    all_aucs = []
    all_specs = []
    spec_types = []
    for spec_type in all_spec_types:
        no_audio_f = 'no_audio_classif_scores_{}.pickle'.format(spec_type)
        with open(os.path.join(results_dir, no_audio_f), 'rb') as f:
            all_specs_na, _, _, aucs_na, _, _ = pickle.load(f)
        
        all_spec_names = [s.comm_name for s in all_specs_na]
            
        fname = 'classif_scores_sorted_{}_{}_{}.pickle'.format(feat, score_type, spec_type)
        load_path = os.path.join(results_dir, fname)
        with open(load_path, 'rb') as f:
            all_specs_st, _, _, aucs_st, auc_ks_st, _, _, _, _, _, _ = pickle.load(f)
        
        spec_names_st = [s.comm_name for s in all_specs_st]
        
        match_names, match_ixs_1, match_ixs_2 = np.intersect1d(spec_names_st, all_spec_names,return_indices=True)
        all_aucs.extend(aucs_st[match_ixs_2])
        all_aucs_na.extend(aucs_na[match_ixs_1])
        all_specs.extend(all_specs_na[match_ixs_1])
        spec_types.extend([spec_type] * len(all_specs_na[match_ixs_1]))
        
        
        print('{} mean inc {}'.format(spec_type, np.mean(aucs_st[match_ixs_2]) - np.mean(aucs_na[match_ixs_1])))
                
        print('{} num inc {}'.format(spec_type, np.sum(aucs_st[match_ixs_2]>aucs_na[match_ixs_1])))
        
    all_aucs = np.asarray(all_aucs)
    all_aucs_na = np.asarray(all_aucs_na)
    
    
    tt_stat, tt_p = scipy.stats.ttest_rel(all_aucs_na, all_aucs)
    mean_inc = np.mean(all_aucs) - np.mean(all_aucs_na)
    num_inc = np.sum(all_aucs>all_aucs_na)
    print('T test: stat = {}, p = {}. Mean inc = {} num inc = {}'.format(tt_stat,tt_p,mean_inc,num_inc))

    
    plt_mat = [all_aucs_na, all_aucs]

    plt_mat_np = copy.copy(plt_mat)
    plt_mat_np = np.asarray(plt_mat_np)
    perc_incs = []
    for s_ix, s_auc in enumerate(plt_mat_np.T):
        spec = all_specs[s_ix]
        st = spec_types[s_ix]

        c = get_spec_type_col(st)

        alpha = 0.8
        lw = 1
        if spec.comm_name.lower().replace(' ','-') in lab_specs:
            alpha = 1
            lw = 2
            plt.text(1.03,s_auc[1],spec.comm_name,verticalalignment='center',color=c)
        plt.plot(s_auc,c=c,alpha=alpha,lw=lw)
        plt.scatter([0,1],s_auc,c=c,alpha=alpha,s=20)
        
        perc_incs.append((s_auc[1]-s_auc[0])/s_auc[0])
    
    print('Mean percent AUC increase: {}'.format(np.mean(perc_incs)))
    
    for ix, data in enumerate(plt_mat):
        c = 'k'
        plt.scatter(ix,np.mean(data),s=300,c=c)
    
        
        h = scipy.stats.sem(data)
        plt.errorbar(ix,np.mean(data),h,c=c,lw=lw,capsize=30,capthick=lw)
        
        #plt.gca().axhline(np.mean(data)+h,c=c,lw=0.5,ls='--')
        #plt.gca().axhline(np.mean(data)-h,c=c,lw=0.5,ls='--')

    #plt.violinplot(plt_mat,positions=range(len(plt_mat)))

    x_buff = 0.4
    plt.xlim([-x_buff,1 + x_buff])
    #plt.gca().set_aspect(1./plt.gca().get_data_ratio())

    plt.ylabel('Species classification AUC\n')
    #plt.xlabel('Model')
    plt.xticks(np.asarray(range(len(plt_mat))),['AGB model', 'Soundscape model'])
    plt.tight_layout()

if __name__ == '__main__':
    matplotlib.rc('font', size=25)
    fig = plt.figure(figsize=(11,8))

    plot_no_audio_comp()

    plt.savefig(os.path.join('figs','fig_4.pdf'),format='pdf')

    plt.show()
    
