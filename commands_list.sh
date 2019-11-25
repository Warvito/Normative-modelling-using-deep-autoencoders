# Initiate virtual enviroment
source venv/bin/activate

# Make all files executable
chmod -R +x ./

# Run python scripts
# ----------------------------- Preprocessing ------------------------------------
# Combining data from different scanners
./combine_sites_data.py -D "ADNI"
./combine_sites_data.py -D "FBF_Brescia"

# Clean UKBiobank data
./clean_biobank1_data.py

# Clean clinical datasets
./clean_clinical_data.py -D "ADNI"
./clean_clinical_data.py -D "FBF_Brescia"
./clean_clinical_data.py -D "OASIS1"

# ------------------------Bootstrap Analysis -------------------------------------
# Create list of ids for bootstrap analysis
./bootstrap_create_ids.py

# Train normative model
./bootstrap_train_aae_supervised.py

# Calculate deviations on clinical data
./bootstrap_test_aae_supervised.py -D "ADNI"
./bootstrap_test_aae_supervised.py -D "FBF_Brescia"
./bootstrap_test_aae_supervised.py -D "OASIS1"

# Perform statistical analysis
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 17
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 27
./bootstrap_group_analysis_1x1.py -D "ADNI" -L 28

./bootstrap_group_analysis_1x1.py -D "FBF_Brescia" -L 17
./bootstrap_group_analysis_1x1.py -D "FBF_Brescia" -L 18

./bootstrap_group_analysis_1x1.py -D "OASIS1" -L 17

# Create Figure 2
./bootstrap_create_figures.py

# ----------------------- Classifier Analysis ------------------------------------
# Create list of ids for classifier analysis
./classifier_create_ids.py -D "ADNI" -L 17
./classifier_create_ids.py -D "ADNI" -L 27
./classifier_create_ids.py -D "ADNI" -L 28

./classifier_create_ids.py -D "FBF_Brescia" -L 17
./classifier_create_ids.py -D "FBF_Brescia" -L 18

./classifier_create_ids.py -D "OASIS1" -L 17

# Train classifiers
./classifier_train.py -D "ADNI" -L 17
./classifier_train.py -D "ADNI" -L 27
./classifier_train.py -D "ADNI" -L 28

./classifier_train.py -D "FBF_Brescia" -L 17
./classifier_train.py -D "FBF_Brescia" -L 18

./classifier_train.py -D "OASIS1" -L 17

# Calculate performance
./classifier_group_analysis_1x1.py -D "ADNI" -L 17
./classifier_group_analysis_1x1.py -D "ADNI" -L 27
./classifier_group_analysis_1x1.py -D "ADNI" -L 28

./classifier_group_analysis_1x1.py -D "FBF_Brescia" -L 17
./classifier_group_analysis_1x1.py -D "FBF_Brescia" -L 18

./classifier_group_analysis_1x1.py -D "OASIS1" -L 17

# --------------------------- Misc -----------------------------------------------
# Perform mass-univariate analysis
./univariate_analysis.py -D "ADNI" -L 17
./univariate_analysis.py -D "ADNI" -L 27
./univariate_analysis.py -D "ADNI" -L 28

./univariate_analysis.py -D "FBF_Brescia" -L 17
./univariate_analysis.py -D "FBF_Brescia" -L 18

./univariate_analysis.py -D "OASIS1" -L 17
