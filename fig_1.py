import os
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
from fig_compare_classif_feats import do_auc_tscale_plot
from fig_chi2_auc_site_time import plot_chi2_fig


'''
Figure 1 is a combination of do_auc_tscale_plot and plot_chi2_fig - this script simply
places the two individual plots into subplots of a larger figure
'''

matplotlib.rc('font', size=18)

fig = plt.figure(figsize=(21,8.5))
gs = fig.add_gridspec(1, 5)

lab_specs = np.asarray(['sooty-capped-babbler', 'tree-hole-frog','rhinoceros-hornbill','bold-striped-tit-babbler'])

lab_all_specs = False
all_spec_types = ['herp','avi','avi-rl']

score_type = 'p60'
feat = 'raw_audioset_feats_3s'

fig.add_subplot(gs[0, :3])

do_auc_tscale_plot(lab_specs, lab_all_specs, all_spec_types, offset_aucs=False, score_type=score_type)

plt.gca().text(-0.07, 1.07, '(a)', transform=plt.gca().transAxes,fontsize=26, va='top', ha='right')

fig.add_subplot(gs[0, 3:])

plot_chi2_fig(lab_specs, lab_all_specs, all_spec_types, score_type=score_type, feat=feat)

plt.gca().text(-0.07, 1.07, '(b)', transform=plt.gca().transAxes,fontsize=26, va='top', ha='right')

plt.tight_layout(pad=3.0)

plt.savefig(os.path.join('figs','fig_2.pdf'),format='pdf')
plt.show()
