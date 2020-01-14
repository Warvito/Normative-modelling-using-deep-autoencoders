## Initiate virtual enviroment
#source venv/bin/activate
#
## Make all files executable
#chmod -R +x ./
#
## Run python scripts
## ----------------------------- Getting data -------------------------------------
## Download data from network-attached storage (MLMH lab use only)
./download_datasets.py -P "/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning/"
#
# Combining data from different scanners
./combine_sites_data.py -D "ADNI2"
./combine_sites_data.py -D "ADNIGO"
./combine_sites_data.py -D "ADNI3"
./combine_sites_data.py -D "TOMC"
./combine_sites_data.py -D "AIBL"
./combine_sites_data.py -D "BIOBANK"

./combine_adni_data.py

# ----------------------------- Preprocessing ------------------------------------
# Clean UK Biobank data
./clean_biobank1_data.py

# Clean clinical datasets
./clean_clinical_data.py -D "ADNI"
./clean_clinical_data.py -D "TOMC"
./clean_clinical_data.py -D "OASIS1"
./clean_clinical_data.py -D "AIBL"
./clean_clinical_data.py -D "MIRIAD"

# Make clinical datasets homogeneous accross age and gender
./demographic_balancing_adni_data.py
./demographic_balancing_tomc_data.py
./demographic_balancing_oasis1_data.py
./demographic_balancing_aibl_data.py
./demographic_balancing_miriad_data.py

# ------------------------Bootstrap Analysis -------------------------------------
# Create list of ids for bootstrap analysis
./bootstrap_create_ids.py

# Train normative model
./bootstrap_train_aae_supervised.py

# Calculate deviations on clinical data
./bootstrap_test_aae_supervised.py -D "ADNI"
./bootstrap_test_aae_supervised.py -D "TOMC"
./bootstrap_test_aae_supervised.py -D "OASIS1"
./bootstrap_test_aae_supervised.py -D "AIBL"
./bootstrap_test_aae_supervised.py -D "MIRIAD"

# Perform statistical analysis
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 17
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 27
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 28

./bootstrap_group_analysis_1x1.py -D "TOMC" -L 17
./bootstrap_group_analysis_1x1.py -D "TOMC" -L 18

./bootstrap_group_analysis_1x1.py -D "OASIS1" -L 17

./bootstrap_group_analysis_1x1.py -D "AIBL" -L 17
./bootstrap_group_analysis_1x1.py -D "AIBL" -L 18

./bootstrap_group_analysis_1x1.py -D "MIRIAD" -L 17

# Create Figure 2
./bootstrap_create_figures.py

# Perform hypothesis test
./bootstrap_hypothesis_test.py -D "ADNI" -L 1 17 27 28
./bootstrap_hypothesis_test.py -D "TOMC" -L 1 17 18
./bootstrap_hypothesis_test.py -D "OASIS1" -L 1 17
./bootstrap_hypothesis_test.py -D "AIBL" -L 1 17 18
./bootstrap_hypothesis_test.py -D "MIRIAD" -L 1 17

# ----------------------- Classifier Analysis ------------------------------------
# Create list of ids for classifier analysis
./classifier_create_ids.py -D "ADNI" -L 17
./classifier_create_ids.py -D "ADNI" -L 27
./classifier_create_ids.py -D "ADNI" -L 28

./classifier_create_ids.py -D "TOMC" -L 17
./classifier_create_ids.py -D "TOMC" -L 18

./classifier_create_ids.py -D "OASIS1" -L 17

./classifier_create_ids.py -D "AIBL" -L 17
./classifier_create_ids.py -D "AIBL" -L 18

./classifier_create_ids.py -D "MIRIAD" -L 17

# Train classifiers
./classifier_train.py -D "ADNI" -L 17
./classifier_train.py -D "ADNI" -L 27
./classifier_train.py -D "ADNI" -L 28

./classifier_train.py -D "TOMC" -L 17
./classifier_train.py -D "TOMC" -L 18

./classifier_train.py -D "OASIS1" -L 17

./classifier_train.py -D "AIBL" -L 17
./classifier_train.py -D "AIBL" -L 18

./classifier_train.py -D "MIRIAD" -L 17

# Calculate performance
./classifier_group_analysis_1x1.py -D "ADNI" -L 17
./classifier_group_analysis_1x1.py -D "ADNI" -L 27
./classifier_group_analysis_1x1.py -D "ADNI" -L 28

./classifier_group_analysis_1x1.py -D "TOMC" -L 17
./classifier_group_analysis_1x1.py -D "TOMC" -L 18

./classifier_group_analysis_1x1.py -D "OASIS1" -L 17

./classifier_group_analysis_1x1.py -D "AIBL" -L 17
./classifier_group_analysis_1x1.py -D "AIBL" -L 18

./classifier_group_analysis_1x1.py -D "MIRIAD" -L 17

# Comparing methods
./classifier_vs_normative.py -D "ADNI" -L 17
./classifier_vs_normative.py -D "ADNI" -L 27
./classifier_vs_normative.py -D "ADNI" -L 28

./classifier_vs_normative.py -D "TOMC" -L 17
./classifier_vs_normative.py -D "TOMC" -L 18

./classifier_vs_normative.py -D "OASIS1" -L 17

./classifier_vs_normative.py -D "AIBL" -L 17
./classifier_vs_normative.py -D "AIBL" -L 18

./classifier_vs_normative.py -D "MIRIAD" -L 17

# Calculate generalization
./classifier_test.py -D "ADNI" -L 17 -E "OASIS1"
./classifier_test.py -D "ADNI" -L 17 -E "TOMC"
./classifier_test.py -D "ADNI" -L 17 -E "MIRIAD"
./classifier_test.py -D "ADNI" -L 17 -E "AIBL"

./classifier_test.py -D "TOMC" -L 17 -E "ADNI"
./classifier_test.py -D "TOMC" -L 17 -E "MIRIAD"
./classifier_test.py -D "TOMC" -L 17 -E "OASIS1"
./classifier_test.py -D "TOMC" -L 17 -E "AIBL"
./classifier_test.py -D "TOMC" -L 18 -E "AIBL"

./classifier_test.py -D "OASIS1" -L 17 -E "ADNI"
./classifier_test.py -D "OASIS1" -L 17 -E "TOMC"
./classifier_test.py -D "OASIS1" -L 17 -E "MIRIAD"
./classifier_test.py -D "OASIS1" -L 17 -E "AIBL"

./classifier_test.py -D "AIBL" -L 17 -E "ADNI"
./classifier_test.py -D "AIBL" -L 17 -E "TOMC"
./classifier_test.py -D "AIBL" -L 17 -E "OASIS1"
./classifier_test.py -D "AIBL" -L 17 -E "AIBL"
./classifier_test.py -D "AIBL" -L 18 -E "TOMC"

./classifier_test.py -D "MIRIAD" -L 17 -E "ADNI"
./classifier_test.py -D "MIRIAD" -L 17 -E "TOMC"
./classifier_test.py -D "MIRIAD" -L 17 -E "OASIS1"
./classifier_test.py -D "MIRIAD" -L 17 -E "AIBL"


./classifier_vs_normative_generalization.py -D "ADNI" -L 17 -E "OASIS1"
./classifier_vs_normative_generalization.py -D "ADNI" -L 17 -E "TOMC"
./classifier_vs_normative_generalization.py -D "ADNI" -L 17 -E "MIRIAD"
./classifier_vs_normative_generalization.py -D "ADNI" -L 17 -E "AIBL"

./classifier_vs_normative_generalization.py -D "TOMC" -L 17 -E "ADNI"
./classifier_vs_normative_generalization.py -D "TOMC" -L 17 -E "MIRIAD"
./classifier_vs_normative_generalization.py -D "TOMC" -L 17 -E "OASIS1"
./classifier_vs_normative_generalization.py -D "TOMC" -L 17 -E "AIBL"
./classifier_vs_normative_generalization.py -D "TOMC" -L 18 -E "AIBL"

./classifier_vs_normative_generalization.py -D "OASIS1" -L 17 -E "ADNI"
./classifier_vs_normative_generalization.py -D "OASIS1" -L 17 -E "TOMC"
./classifier_vs_normative_generalization.py -D "OASIS1" -L 17 -E "MIRIAD"
./classifier_vs_normative_generalization.py -D "OASIS1" -L 17 -E "AIBL"

./classifier_vs_normative_generalization.py -D "AIBL" -L 17 -E "ADNI"
./classifier_vs_normative_generalization.py -D "AIBL" -L 17 -E "TOMC"
./classifier_vs_normative_generalization.py -D "AIBL" -L 17 -E "OASIS1"
./classifier_vs_normative_generalization.py -D "AIBL" -L 17 -E "AIBL"
./classifier_vs_normative_generalization.py -D "AIBL" -L 18 -E "TOMC"

./classifier_vs_normative_generalization.py -D "MIRIAD" -L 17 -E "ADNI"
./classifier_vs_normative_generalization.py -D "MIRIAD" -L 17 -E "TOMC"
./classifier_vs_normative_generalization.py -D "MIRIAD" -L 17 -E "OASIS1"
./classifier_vs_normative_generalization.py -D "MIRIAD" -L 17 -E "AIBL"

# --------------------------- Misc -----------------------------------------------
# Perform mass-univariate analysis
./univariate_analysis.py -D "ADNI" -L 17
./univariate_analysis.py -D "ADNI" -L 27
./univariate_analysis.py -D "ADNI" -L 28

./univariate_analysis.py -D "TOMC" -L 17
./univariate_analysis.py -D "TOMC" -L 18

./univariate_analysis.py -D "OASIS1" -L 17

./univariate_analysis.py -D "AIBL" -L 17
./univariate_analysis.py -D "AIBL" -L 18

./univariate_analysis.py -D "MIRIAD" -L 17