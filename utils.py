"""Helper functions and constants."""
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()


def cliff_delta(X, Y):
    """Calculate the effect size using the Cliff's delta."""
    lx = len(X)
    ly = len(Y)
    mat = np.zeros((lx, ly))
    for i in range(0, lx):
        for j in range(0, ly):
            if X[i] > Y[j]:
                mat[i, j] = 1
            elif Y[j] > X[i]:
                mat[i, j] = -1

    return (np.sum(mat)) / (lx * ly)


def load_dataset(demographic_path, ids_path, freesurfer_path):
    """Load dataset."""
    dataset = load_demographic_data(demographic_path, ids_path)

    # Loading Freesurfer data
    freesurfer = pd.read_csv(freesurfer_path)

    # Create a new col in FS dataset to contain Participant_ID
    if 'FBF_Brescia' in str(demographic_path):
        freesurfer['Participant_ID'] = freesurfer['Image_ID'].str.split('_').str[0] + '_' + \
                                       freesurfer['Image_ID'].str.split('_').str[1]
    else:
        freesurfer['Participant_ID'] = freesurfer['Image_ID'].str.split('_', expand=True)[0]

    # Merge FS dataset and demographic dataset to access age
    dataset = pd.merge(freesurfer, dataset, on='Participant_ID')

    if 'Image_ID_y' in dataset.columns:
        dataset['Image_ID'] = dataset['Image_ID_x']
        dataset = dataset.drop(['Image_ID_y', 'Image_ID_x'], axis=1)

    return dataset


def load_demographic_data(demographic_path, ids_path):
    """Load dataset using selected ids."""

    if demographic_path.suffix == '.tsv':
        demographic_df = pd.read_csv(demographic_path, sep='\t')
    else:
        demographic_df = pd.read_csv(demographic_path)

    # if using UK Biobank supplementary data
    if 'eid' in demographic_df.columns:
        demographic_df = demographic_df[['eid', '31-0.0', '21000-0.0', '21003-2.0']]
        demographic_df.columns = ['ID', 'Gender', 'Ethnicity', 'Age']

        # Labeling data
        ethnicity_dict = {
            1: 'White', 1001: 'White', 1002: 'White', 1003: 'White',
            2: 'Mixed', 2001: 'Mixed', 2002: 'Mixed', 2003: 'Mixed', 2004: 'Mixed',
            3: 'Asian', 3001: 'Asian', 3002: 'Asian', 3003: 'Asian', 3004: 'Asian',
            4: 'Black', 4001: 'Black', 4002: 'Black', 4003: 'Black',
            5: 'Chinese',
            6: 'Other',
            -1: 'Not known', -3: 'Not known'
        }

        gender_dict = {
            0: 'Female',
            1: 'Male'
        }

        demographic_df = demographic_df.replace({'Gender': gender_dict})
        demographic_df = demographic_df.replace({'Ethnicity': ethnicity_dict})

    # if using participants.tsv file
    else:
        demographic_df['ID'] = demographic_df['Participant_ID'].str.split('-').str[1]

    # demographic_df['ID'] = pd.to_numeric(demographic_df['ID'])
    demographic_df = demographic_df.dropna()

    ids_df = pd.read_csv(ids_path)
    # Create a new 'ID' column to match supplementary demographic data
    if 'Participant_ID' in ids_df.columns:
        # For create_homogeneous_data.py output
        ids_df['ID'] = ids_df['Participant_ID'].str.split('-').str[1]
    elif 'FBF_Brescia' in str(demographic_path):
        ids_df['ID'] = ids_df['Image_ID'].str.split('_').str[0] + '_' + ids_df['Image_ID'].str.split('_').str[1]
        ids_df['ID'] = ids_df['ID'].str.split('-').str[1]
    else:
        # For freesurferData dataframe
        ids_df['ID'] = ids_df['Image_ID'].str.split('_').str[0]
        ids_df['ID'] = ids_df['ID'].str.split('-').str[1]

    # ids_df['ID'] = pd.to_numeric(ids_df['ID'])

    # Merge supplementary demographic data with ids
    demographic_df['ID'] = demographic_df['ID'].astype('int')
    ids_df['ID'] = ids_df['ID'].astype('int')

    dataset = pd.merge(ids_df, demographic_df, on='ID')

    if 'Participant_ID_y' in dataset.columns:
        dataset['Participant_ID'] = dataset['Participant_ID_x']
        dataset = dataset.drop(['Participant_ID_x', 'Participant_ID_y'], axis=1)

    if 'Image_ID_y' in dataset.columns:
        dataset['Image_ID'] = dataset['Image_ID_x']
        dataset = dataset.drop(['Image_ID_y', 'Image_ID_x'], axis=1)

    return dataset


COLUMNS_NAME = ['Left-Lateral-Ventricle',
                'Left-Inf-Lat-Vent',
                'Left-Cerebellum-White-Matter',
                'Left-Cerebellum-Cortex',
                'Left-Thalamus-Proper',
                'Left-Caudate',
                'Left-Putamen',
                'Left-Pallidum',
                '3rd-Ventricle',
                '4th-Ventricle',
                'Brain-Stem',
                'Left-Hippocampus',
                'Left-Amygdala',
                'CSF',
                'Left-Accumbens-area',
                'Left-VentralDC',
                'Right-Lateral-Ventricle',
                'Right-Inf-Lat-Vent',
                'Right-Cerebellum-White-Matter',
                'Right-Cerebellum-Cortex',
                'Right-Thalamus-Proper',
                'Right-Caudate',
                'Right-Putamen',
                'Right-Pallidum',
                'Right-Hippocampus',
                'Right-Amygdala',
                'Right-Accumbens-area',
                'Right-VentralDC',
                'CC_Posterior',
                'CC_Mid_Posterior',
                'CC_Central',
                'CC_Mid_Anterior',
                'CC_Anterior',
                'lh_bankssts_volume',
                'lh_caudalanteriorcingulate_volume',
                'lh_caudalmiddlefrontal_volume',
                'lh_cuneus_volume',
                'lh_entorhinal_volume',
                'lh_fusiform_volume',
                'lh_inferiorparietal_volume',
                'lh_inferiortemporal_volume',
                'lh_isthmuscingulate_volume',
                'lh_lateraloccipital_volume',
                'lh_lateralorbitofrontal_volume',
                'lh_lingual_volume',
                'lh_medialorbitofrontal_volume',
                'lh_middletemporal_volume',
                'lh_parahippocampal_volume',
                'lh_paracentral_volume',
                'lh_parsopercularis_volume',
                'lh_parsorbitalis_volume',
                'lh_parstriangularis_volume',
                'lh_pericalcarine_volume',
                'lh_postcentral_volume',
                'lh_posteriorcingulate_volume',
                'lh_precentral_volume',
                'lh_precuneus_volume',
                'lh_rostralanteriorcingulate_volume',
                'lh_rostralmiddlefrontal_volume',
                'lh_superiorfrontal_volume',
                'lh_superiorparietal_volume',
                'lh_superiortemporal_volume',
                'lh_supramarginal_volume',
                'lh_frontalpole_volume',
                'lh_temporalpole_volume',
                'lh_transversetemporal_volume',
                'lh_insula_volume',
                'rh_bankssts_volume',
                'rh_caudalanteriorcingulate_volume',
                'rh_caudalmiddlefrontal_volume',
                'rh_cuneus_volume',
                'rh_entorhinal_volume',
                'rh_fusiform_volume',
                'rh_inferiorparietal_volume',
                'rh_inferiortemporal_volume',
                'rh_isthmuscingulate_volume',
                'rh_lateraloccipital_volume',
                'rh_lateralorbitofrontal_volume',
                'rh_lingual_volume',
                'rh_medialorbitofrontal_volume',
                'rh_middletemporal_volume',
                'rh_parahippocampal_volume',
                'rh_paracentral_volume',
                'rh_parsopercularis_volume',
                'rh_parsorbitalis_volume',
                'rh_parstriangularis_volume',
                'rh_pericalcarine_volume',
                'rh_postcentral_volume',
                'rh_posteriorcingulate_volume',
                'rh_precentral_volume',
                'rh_precuneus_volume',
                'rh_rostralanteriorcingulate_volume',
                'rh_rostralmiddlefrontal_volume',
                'rh_superiorfrontal_volume',
                'rh_superiorparietal_volume',
                'rh_superiortemporal_volume',
                'rh_supramarginal_volume',
                'rh_frontalpole_volume',
                'rh_temporalpole_volume',
                'rh_transversetemporal_volume',
                'rh_insula_volume']

DATASETS_REPLACE_DICT = {
    'BIOBANK-SCANNER01': 0,
    'SCANNER002': 1,
    'SCANNER003': 1,
    'SCANNER006': 1,
    'SCANNER009': 1,
    'SCANNER011': 1,
    'SCANNER012': 1,
    'SCANNER013': 1,
    'SCANNER014': 1,
    'SCANNER018': 1,
    'SCANNER019': 1,
    'SCANNER020': 1,
    'SCANNER022': 1,
    'SCANNER023': 1,
    'SCANNER024': 1,
    'SCANNER031': 1,
    'SCANNER032': 1,
    'SCANNER033': 1,
    'SCANNER035': 1,
    'SCANNER036': 1,
    'SCANNER037': 1,
    'SCANNER041': 1,
    'SCANNER053': 1,
    'SCANNER067': 1,
    'SCANNER068': 1,
    'SCANNER070': 1,
    'SCANNER072': 1,
    'SCANNER073': 1,
    'SCANNER082': 1,
    'SCANNER094': 1,
    'SCANNER099': 1,
    'SCANNER100': 1,
    'SCANNER116': 1,
    'SCANNER123': 1,
    'SCANNER128': 1,
    'SCANNER130': 1,
    'SCANNER131': 1,
    'SCANNER135': 1,
    'SCANNER136': 1,
    'SCANNER137': 1,
    'SCANNER153': 1,
    'SCANNER941': 1,
    'SCANNER141': 1,
    'SCANNER051': 1,
    'SCANNER114': 1,
    'SCANNER098': 1,
    'SCANNER057': 1,
    'FBF_Brescia-SCANNER01': 2,
    'FBF_Brescia-SCANNER02': 2,
    'FBF_Brescia-SCANNER03': 2}
