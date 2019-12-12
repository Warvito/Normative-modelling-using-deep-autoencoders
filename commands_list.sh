# Initiate virtual enviroment
source venv/bin/activate

# Make all files executable
chmod -R +x ./

# Run python scripts
# ----------------------------- Preprocessing ------------------------------------
# Download data from network-attached storage (MLMH lab use only)
./download_datasets.py

# Combining data from different scanners
./combine_sites_data.py -D "ADNI"
./combine_sites_data.py -D "TOMC"
./combine_sites_data.py -D "BIOBANK"

# Clean UKBiobank data
./clean_biobank1_data.py

# Clean clinical datasets
./clean_clinical_data.py -D "ADNI"
./clean_clinical_data.py -D "TOMC"
./clean_clinical_data.py -D "OASIS1"

# Make clinical datasets homogeneous accross age and gender
./demographic_balancing_adni_data.py
./demographic_balancing_tomc_data.py
./demographic_balancing_oasis1_data.py

# ------------------------Bootstrap Analysis -------------------------------------
# Create list of ids for bootstrap analysis
./bootstrap_create_ids.py

# Train normative model
./bootstrap_train_aae_supervised.py

# Calculate deviations on clinical data
./bootstrap_test_aae_supervised.py -D "ADNI"
./bootstrap_test_aae_supervised.py -D "TOMC"
./bootstrap_test_aae_supervised.py -D "OASIS1"

# Perform statistical analysis
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 17
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 27
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 28

./bootstrap_group_analysis_1x1.py -D "TOMC" -L 17
./bootstrap_group_analysis_1x1.py -D "TOMC" -L 18

./bootstrap_group_analysis_1x1.py -D "OASIS1" -L 17

# Create Figure 2
./bootstrap_create_figures.py

# Perform hypothesis test
./bootstrap_hypothesis_test.py -D "ADNI" -L 1 17 27 28
./bootstrap_hypothesis_test.py -D "TOMC" -L 1 17 18
./bootstrap_hypothesis_test.py -D "OASIS1" -L 1 17

# ----------------------- Classifier Analysis ------------------------------------
# Create list of ids for classifier analysis
./classifier_create_ids.py -D "ADNI" -L 17
./classifier_create_ids.py -D "ADNI" -L 27
./classifier_create_ids.py -D "ADNI" -L 28

./classifier_create_ids.py -D "TOMC" -L 17
./classifier_create_ids.py -D "TOMC" -L 18

./classifier_create_ids.py -D "OASIS1" -L 17

# Train classifiers
./classifier_train.py -D "ADNI" -L 17
./classifier_train.py -D "ADNI" -L 27
./classifier_train.py -D "ADNI" -L 28

./classifier_train.py -D "TOMC" -L 17
./classifier_train.py -D "TOMC" -L 18

./classifier_train.py -D "OASIS1" -L 17

# Calculate performance
./classifier_group_analysis_1x1.py -D "ADNI" -L 17
./classifier_group_analysis_1x1.py -D "ADNI" -L 27
./classifier_group_analysis_1x1.py -D "ADNI" -L 28

./classifier_group_analysis_1x1.py -D "TOMC" -L 17
./classifier_group_analysis_1x1.py -D "TOMC" -L 18

./classifier_group_analysis_1x1.py -D "OASIS1" -L 17

# comparing methods
./bootstrap_normative_vs_classifier.py -D "ADNI" -L 17
./bootstrap_normative_vs_classifier.py -D "ADNI" -L 27
./bootstrap_normative_vs_classifier.py -D "ADNI" -L 28

./bootstrap_normative_vs_classifier.py -D "TOMC" -L 17
./bootstrap_normative_vs_classifier.py -D "TOMC" -L 18

./bootstrap_normative_vs_classifier.py -D "OASIS1" -L 17

# --------------------------- Misc -----------------------------------------------
# Perform mass-univariate analysis
./univariate_analysis.py -D "ADNI" -L 17
./univariate_analysis.py -D "ADNI" -L 27
./univariate_analysis.py -D "ADNI" -L 28

./univariate_analysis.py -D "TOMC" -L 17
./univariate_analysis.py -D "TOMC" -L 18

./univariate_analysis.py -D "OASIS1" -L 17
