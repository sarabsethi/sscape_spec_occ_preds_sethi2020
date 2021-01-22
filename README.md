# sscape_spec_occ_preds_sethi2020
Reproducability code for "Soundscapes predict species occurrence in tropical forests"

More detailed instructions will follow in the future, but the following instructions should allow you to get started (assumes Python 3.7):

## Run analyses

* Download the associated data from Zenodo, and unzip it into the directory `pc_data`
* Run `python analysis_classif_species_logo.py` to perform the K-fold classification task which predicts species occcurence from audio features
  * Fitting GMMs to audio features on this scale is computationally expensive and is typically run on HPC facilities (the script is set up to work using Imperial's infrastructure, but will also work slowly running locally on a single machine)
* Run `python analysis_classif_k_aucs.py` to boil the K-fold classification results down to summary statistics
* Run `python analysis_classif_no_audio.py` to perform a similar K-fold classification task but using AGB instead of audio
 
## Reproduce figures from "Soundscapes predict species occurrence in tropical forests"

* Fig. 1: `python fig_1.py`
* Fig. 2: `python fig_classif_surface_site_time.py` with variable in script `show_all_specs = False`
* Fig. 3: `python fig_compare_no_audio.py`

* Fig. S2: `python fig_compare_classif_score_types.py`
* Fig. S3: `python fig_within_pc_llhood_ratios.py`
* Fig. S4: `python fig_n_occs_auc_corr.py` 
* Fig. S5: `python fig_classif_surface_site_time.py` with variable in script `show_all_specs = True`
* Fig. S6: `python fig_auc_by_site.py`
