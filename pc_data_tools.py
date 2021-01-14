import pickle
import os
import numpy as np
import scipy.stats as stats
import pc_class_defs
from tqdm import tqdm

PC_PICKLE_SAVE_DIR = 'pc_data'
SIS_RES_PICKLE_SAVE_DIR = 'results_sis'
CLASS_RES_PICKLE_SAVE_DIR = 'results_classifications_logo'


def get_nice_lab(ugly_lab):
    '''
    Get nice labels for plotting using shorthands used throughout code
    '''

    # Species types
    if ugly_lab == 'avi': return 'Avifaunal species'
    if ugly_lab == 'avi-rl': return 'Avifaunal species (threatened - IUCN Red List)'
    if ugly_lab == 'herp': return 'Herpetofaunal species'

    # Score types
    if ugly_lab == 'min': return 'Min'
    if ugly_lab == 'mean': return 'Mean'
    if ugly_lab == 'max': return 'Max'
    if ugly_lab == 'mean+max': return 'Mean + Max'

    # Percentiles
    if ugly_lab[0] == 'p' and len(ugly_lab) == 3:
        ptile = ugly_lab[1:]
        return r'$P_{' + ptile + '}$'

    # If nothing found, just return the argument
    return ugly_lab

def get_spec_type_col(spec_type):
    '''
    Get colours to represent different species types
    '''

    if spec_type == 'avi':
        return '#2782D2'
    if spec_type == 'herp':
        return '#45ad44'
    if spec_type == 'avi-rl':
        return '#960303'

    return '#000000'

def save_pc_dataset(all_sites, all_taxa, all_pcs, save_fname):
    '''
    Save point count dataset to a pickle file
    '''

    if not os.path.exists(PC_PICKLE_SAVE_DIR): os.makedirs(PC_PICKLE_SAVE_DIR)

    save_path = os.path.join(PC_PICKLE_SAVE_DIR, 'pc_data_parsed_{}.pickle'.format(save_fname))
    with open(save_path, 'wb') as f:
        pickle.dump([all_sites, all_taxa, all_pcs], f)

def load_pc_dataset(load_f_name):
    '''
    Load a point count dataset from a pickle file
    '''

    if not load_f_name.lower().endswith('.pickle'): load_f_name = load_f_name + '.pickle'

    load_path = os.path.join(PC_PICKLE_SAVE_DIR, load_f_name)

    with open(load_path, 'rb') as f:
        all_sites, all_taxa, all_pcs = pickle.load(f)

    # Make sure the same features are linked to all the point counts
    all_feat_names = [pc.audio_feat_name for pc in all_pcs]
    feat_names_consistent = all([feat_name == all_feat_names[0] for feat_name in all_feat_names])
    if feat_names_consistent: tqdm.write('All feature names are consistent across PCs: {}'.format(all_feat_names[0]))
    else: raise Exception('Not all feature names are consistent across PCs')

    # Only return point counts which have audio features linked
    all_pcs_w_feats = [pc for pc in all_pcs if pc.audio_feats.shape[0] > 0]

    return all_feat_names[0], all_sites, all_taxa, all_pcs_w_feats


def save_classification_data(species, chosen_pcs, train_index, test_index, pcs_feats, pcs_spec_labs, pres_gmm_train, abs_gmm_train, save_fname, spec_type, save_dir = CLASS_RES_PICKLE_SAVE_DIR):
    '''
    Save results from the classification task
    '''

    save_dir = '{}_{}'.format(save_dir,spec_type)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    save_path = os.path.join(save_dir, '{}.pickle'.format(save_fname))
    with open(save_path, 'wb') as f:
        pickle.dump([species, chosen_pcs, train_index, test_index, pcs_feats, pcs_spec_labs, pres_gmm_train, abs_gmm_train], f)

def load_classification_data(load_f_name, load_f_dir=CLASS_RES_PICKLE_SAVE_DIR):
    '''
    Load results from the classification task
    '''

    if not load_f_name.lower().endswith('.pickle'): load_f_name = load_f_name + '.pickle'

    if '/' in load_f_name:
        load_path = load_f_name
    else:
        load_path = os.path.join(load_f_dir, load_f_name)

    with open(load_path, 'rb') as f:
        return pickle.load(f)

def load_pc_spectro(pc_aud_fname):
    '''
    Load precomputed spectrograms linked to a point count audio file
    '''

    load_path = os.path.join('pc_spectros','{}_spectro.pickle'.format(pc_aud_fname))

    with open(load_path, 'rb') as f:
        return pickle.load(f)

def get_pcs_no_water(pc_list):
    '''
    Parse a list of all point counts and filter out those that were from sites near
    water, or those with dud recordings (hard-coded)
    '''

    dud_files = ['SY0000_0058','SY0000_0059','SY0000_0060','SY0000_0061','SY0000_0062','SY0000_0063','PC0000_0019','PC0001_1058','PC0001_1059','PC0001_0155']
    pcs_not_dud = [pc for pc in pc_list if pc.audio_fname not in dud_files]

    skip_sites = ['Riparian 1','Riparian 2','LFE river','B1 602']
    pcs_no_water = [pc for pc in pcs_not_dud if pc.site.name not in skip_sites]

    #print('Removed sites {} to get {}/{} PCs'.format(skip_sites, len(pcs_no_water), len(pc_list)))
    return pcs_no_water

def get_avi_pcs_no_water_sites(pc_list, site_list):
    '''
    Filter get_pcs_no_water for only avifaunal point counts
    '''

    pcs_no_water = get_pcs_no_water(pc_list)
    avi_pcs_no_water = [pc for pc in pcs_no_water if pc.avi]
    #tqdm.write('{} avifaunal PCs, excluding water sites'.format(len(avi_pcs_no_water)))
    return avi_pcs_no_water

def get_herp_pcs_no_water_sites(pc_list, site_list):
    '''
    Filter get_pcs_no_water for only herpetofaunal point counts
    '''

    pcs_no_water = get_pcs_no_water(pc_list)
    herp_pcs_no_water = [pc for pc in pcs_no_water if pc.herp]
    #tqdm.write('{} herpetofaunal PCs, excluding water sites'.format(len(herp_pcs_no_water)))
    return herp_pcs_no_water

def get_avi_specs_min_pres(taxa_list, pc_list, min_pres=51):
    '''
    Filter for avifaunal species which are present in at least <min_pres> point counts
    '''

    avi_pcs_no_water = [pc for pc in pc_list if pc.avi]

    avi_taxa = [t for t in taxa_list if t.is_avi]
    avi_specs = [at for at in avi_taxa if at.rank.lower() == 'species']

    print('Total avi specs (no PC thresh): {}'.format(len(avi_specs)))
    avi_specs_min_present = [at for at in avi_specs if len([pc for pc in avi_pcs_no_water if at in pc.avi_spec_comm]) >= min_pres]

    return avi_specs_min_present

def get_herp_specs_min_pres(taxa_list, pc_list, min_pres=51):
    '''
    Filter for herpetofaunal species which are present in at least <min_pres> point counts
    '''

    herp_pcs_no_water = [pc for pc in pc_list if pc.herp]

    herp_taxa = [t for t in taxa_list if t.is_herp]
    herp_specs = [ht for ht in herp_taxa if ht.rank.lower() == 'species']

    print('Total herp specs (no PC thresh): {}'.format(len(herp_specs)))
    herp_specs_min_present = [ht for ht in herp_specs if len([pc for pc in herp_pcs_no_water if ht in pc.herp_spec_comm]) >= min_pres]

    return herp_specs_min_present

def get_red_list_avi_specs_min_pres(taxa_list, pc_list, min_pres=15):
    '''
    Return avifaunal species which are present in at least <min_pres> point counts
    and belong to the EN, CR, or VU categories in the IUCN RED list
    '''

    avi_pcs_no_water = [pc for pc in pc_list if pc.avi]

    avi_taxa = [t for t in taxa_list if t.is_avi]
    avi_taxa_min_present = [at for at in avi_taxa if len([pc for pc in avi_pcs_no_water if at in pc.avi_spec_comm]) >= min_pres]
    avi_specs_min_present = [at for at in avi_taxa_min_present if at.rank.lower() == 'species']
    avi_specs_min_present_rl = [at for at in avi_specs_min_present if at.get_red_list_status() in ['EN','CR','VU']]

    return avi_specs_min_present_rl

def get_all_class_file_paths(string_filters, all_res_files = None, k=0, classif_res_dir = CLASS_RES_PICKLE_SAVE_DIR, spec_type='avi'):
    '''
    Get a list of all classification results which match the given arguments
    '''

    classif_res_dir = '{}_{}'.format(classif_res_dir, spec_type)

    if all_res_files is None: all_res_files = os.listdir(classif_res_dir)

    if isinstance(string_filters,str):
        string_filters = [string_filters]

    filt_res_files = all_res_files
    for filt in string_filters:
        filt_res_files = [f for f in filt_res_files if filt in f]

    # If k = -1, then return all results from all K folds
    if k != -1:
        filt_res_files = [f for f in filt_res_files if 'k{}'.format(k) in f]

    filt_res_paths = [os.path.join(classif_res_dir,f) for f in filt_res_files if classif_res_dir not in f]

    if len(filt_res_paths) == 0:
        print('No results files found for string_filters {}'.format(string_filters))
        exit()

    return sorted(filt_res_paths)

def get_llhood_ratio_score(feat_samps, pres_gmm, abs_gmm, score_type, score_samps=None):
    '''
    Return a classification score (confidence of a species being present) from a list
    of CNN-derived audio features.

    score_type determines how an array of likelihood ratios is converted into a final
    classification confidence for a whole audio recording
    '''

    if score_samps is None:
        score_samps = pres_gmm.score_samples(feat_samps) - abs_gmm.score_samples(feat_samps)

    ptile_score_types = ['p10','p20','p30','p40','p50','p60','p70','p80','p90','p95']

    if score_type == 'max':
        return np.max(score_samps)
    elif score_type == 'min':
        return np.min(score_samps)
    elif score_type in ptile_score_types:
        ptile = int(score_type[1:])
        return np.percentile(score_samps, ptile)
    elif score_type == 'mean':
        return np.mean(score_samps)
    elif score_type == 'mean+max':
        return np.mean(score_samps) + np.max(score_samps)


def get_secs_per_audio_feat(feat_name):
    '''
    From an audio feature name (e.g. raw_audioset_feats_3s) return how many seconds
    that audio feature represents.

    Note: raw_audioset_feats_3s actually corresponds to 2.88s (the naming is misleading)
    but this function accounts for this and returns the correct value (2.88 in this case)
    '''

    target_secs_per_feat = feat_name.split('_')[-1].split('s')[0]
    if target_secs_per_feat.startswith('0'): target_secs_per_feat = float(target_secs_per_feat)/100
    else: target_secs_per_feat = float(target_secs_per_feat)

    actual_secs_per_feat = 0.96 * int(target_secs_per_feat / 0.96)

    return round(actual_secs_per_feat,2)

def get_nparray_from_feats(feats_list):
    '''
    Convert a list of AudioFeats to an nparray
    '''

    if type(feats_list) is not list:
        feats_list = list(feats_list)

    list_feats = [f.feat_vec for f in feats_list]
    return np.vstack(list_feats)

def unpack_feats_from_pcs(pc_list, only_af_prop_centres=False):
    '''
    Combine all audio features from a list of point counts into one nparray

    If only_af_prop_centres is set, this assumes the clustering step has already been run on the PointCount objects
    '''

    if only_af_prop_centres:
        tqdm.write('Only using features from cluster centres (afin prop) from PC audio data')
    else:
        tqdm.write('Using all features from PC audio data')

    all_afs = []
    for pc in pc_list:
        if only_af_prop_centres:
            pc_feats = pc.af_prop_clust.cluster_centers_
        else:
            pc_feats = pc.audio_feats

        for f_ix, feat in enumerate(pc_feats):
            offs_s = f_ix * pc.secs_per_audio_feat
            all_afs.append(pc_class_defs.AudioFeat(feat,pc,offs_s))

    return np.asarray(all_afs)


def smooth_density_curve(data, interp_pts=1000):
    """
    Create a smoothed density curve from a vector of observations of 1D features

    Args:
        data (ndarray): vector of data
        interp_pts (int): number of points to interpolate between

    Returns:
        x (ndarray): x axis of smoothed distribution
        smoothed_curve (ndarray): y axis of smoothed distribution
    """

    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(),data.max(), interp_pts)
    smoothed_curve = kde(x)

    return x, smoothed_curve
