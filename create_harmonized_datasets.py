"""Script to create datasets harmonized."""
from pathlib import Path

import pandas as pd
from neurocombat_sklearn import CombatModel

from utils import load_dataset, COLUMNS_NAME, DATASETS_REPLACE_DICT

PROJECT_ROOT = Path.cwd()


def harmonize_df(df, model):
    df_harmonized = df.copy()

    df_harmonized[COLUMNS_NAME] = model.transform(df[COLUMNS_NAME],
                                                  df[['Dataset']],
                                                  df[['Gender']],
                                                  df[['Age']])

    df_harmonized = df_harmonized[COLUMNS_NAME + ['Image_ID', 'EstimatedTotalIntraCranialVol']]
    return df_harmonized

def normalize_by_TIV(df):
    for brain_region in COLUMNS_NAME:
        df[brain_region] = df[brain_region] / df['EstimatedTotalIntraCranialVol']
    return df


def main():
    """"""
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'

    biobank_participants_path = PROJECT_ROOT / 'data' / 'datasets' / 'BIOBANK' / 'participants.tsv'
    biobank_freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / 'BIOBANK' / 'freesurferData.csv'
    biobank_ids_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids.csv'

    adni_participants_path = PROJECT_ROOT / 'data' / 'datasets' / 'ADNI' / 'participants.tsv'
    adni_freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / 'ADNI' / 'freesurferData.csv'

    brescia_participants_path = PROJECT_ROOT / 'data' / 'datasets' / 'FBF_Brescia' / 'participants.tsv'
    brescia_freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / 'FBF_Brescia' / 'freesurferData.csv'

    hc_label = 1

    # ----------------------------------------------------------------------------
    # Create directories structure
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    adni_ids_path = experiment_dir / 'ADNI_homogeneous_ids.csv'
    brescia_ids_path = experiment_dir / 'FBF_Brescia_homogeneous_ids.csv'

    adni_harmonized_path = experiment_dir / 'ADNI_harmonizedFreesurferData.csv'
    brescia_harmonized_path = experiment_dir / 'FBF_Brescia_harmonizedFreesurferData.csv'
    biobank_harmonized_path = experiment_dir / 'BIOBANK_harmonizedFreesurferData.csv'

    adni_harmonized_training_ids_path = experiment_dir / 'ADNI_harmonized_training_ids.csv'
    brescia_harmonized_training_ids_path = experiment_dir / 'FBF_Brescia_harmonized_training_ids.csv'

    adni_harmonized_test_ids_path = experiment_dir / 'ADNI_harmonized_test_ids.csv'
    brescia_harmonized_test_ids_path = experiment_dir / 'FBF_Brescia_harmonized_test_ids.csv'

    # ----------------------------------------------------------------------------
    # Loading data
    biobank_dataset_df = load_dataset(biobank_participants_path, biobank_ids_path, biobank_freesurfer_path)

    adni_dataset_df = load_dataset(adni_participants_path, adni_ids_path, adni_freesurfer_path)
    brescia_dataset_df = load_dataset(brescia_participants_path, brescia_ids_path, brescia_freesurfer_path)

    # ----------------------------------------------------------------------------
    # Normalise brain regions by TIV
    biobank_dataset_df = normalize_by_TIV(biobank_dataset_df)
    adni_dataset_df = normalize_by_TIV(adni_dataset_df)
    brescia_dataset_df = normalize_by_TIV(brescia_dataset_df)

    # ----------------------------------------------------------------------------
    # Replace Dataset labels
    biobank_dataset_df = biobank_dataset_df.replace({'Dataset': DATASETS_REPLACE_DICT})
    adni_dataset_df = adni_dataset_df.replace({'Dataset': DATASETS_REPLACE_DICT})
    brescia_dataset_df = brescia_dataset_df.replace({'Dataset': DATASETS_REPLACE_DICT})

    # ----------------------------------------------------------------------------
    # Get healthy controls
    adni_dataset_hc_df = adni_dataset_df.loc[(adni_dataset_df['Diagn'] == hc_label)]
    brescia_dataset_hc_df = brescia_dataset_df.loc[(brescia_dataset_df['Diagn'] == hc_label)]

    # ----------------------------------------------------------------------------
    # Sampling training
    adni_training_df = adni_dataset_hc_df.sample(16, random_state=42)
    adni_test_df = adni_dataset_df[~adni_dataset_df.index.isin(adni_training_df.index)]

    brescia_training_df = brescia_dataset_hc_df.sample(55, random_state=42)
    brescia_test_df = brescia_dataset_df[~brescia_dataset_df.index.isin(brescia_training_df.index)]

    merged_df = pd.concat([biobank_dataset_df,
                           adni_training_df,
                           brescia_training_df])

    model = CombatModel()
    model.fit(merged_df[COLUMNS_NAME],
              merged_df[['Dataset']],
              merged_df[['Gender']],
              merged_df[['Age']])

    adni_freesurfer_harmonized = harmonize_df(adni_dataset_df, model)
    brescia_freesurfer_harmonized = harmonize_df(brescia_dataset_df, model)
    biobank_freesurfer_harmonized = harmonize_df(biobank_dataset_df, model)

    adni_freesurfer_harmonized.to_csv(adni_harmonized_path, index=False)
    brescia_freesurfer_harmonized.to_csv(brescia_harmonized_path, index=False)
    biobank_freesurfer_harmonized.to_csv(biobank_harmonized_path, index=False)

    pd.DataFrame(adni_training_df['Participant_ID']).to_csv(adni_harmonized_training_ids_path, index=False)
    pd.DataFrame(brescia_training_df['Participant_ID']).to_csv(brescia_harmonized_training_ids_path, index=False)

    pd.DataFrame(adni_test_df['Participant_ID']).to_csv(adni_harmonized_test_ids_path, index=False)
    pd.DataFrame(brescia_test_df['Participant_ID']).to_csv(brescia_harmonized_test_ids_path, index=False)



if __name__ == "__main__":
    main()
