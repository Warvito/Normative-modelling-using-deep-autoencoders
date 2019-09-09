"""Script to create datasets harmonized."""
from pathlib import Path

import pandas as pd
from neurocombat_sklearn import CombatModel

from utils import load_dataset, COLUMNS_NAME

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    model_name = 'supervised_aae'

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
    model_dir = experiment_dir / model_name
    model_dir.mkdir(exist_ok=True)

    adni_harmonized_path = experiment_dir / 'ADNI_harmonizedFreesurferData.csv'
    brescia_harmonized_path = experiment_dir / 'FBF_Brescia_harmonizedFreesurferData.csv'
    biobank_harmonized_path = experiment_dir / 'BIOBANK_harmonizedFreesurferData.csv'


    adni_ids_path = experiment_dir / 'ADNI_homogeneous_ids.csv'
    brescia_ids_path = experiment_dir / 'FBF_Brescia_homogeneous_ids.csv'

    # ----------------------------------------------------------------------------
    # Loading data
    biobank_dataset_df = load_dataset(biobank_participants_path, biobank_ids_path, biobank_freesurfer_path)

    adni_dataset_df = load_dataset(adni_participants_path, adni_ids_path, adni_freesurfer_path)
    brescia_dataset_df = load_dataset(brescia_participants_path, brescia_ids_path, brescia_freesurfer_path)

    adni_dataset_hc_df = adni_dataset_df.loc[(adni_dataset_df['Diagn'] == hc_label)]

    brescia_dataset_hc_df = brescia_dataset_df.loc[(brescia_dataset_df['Diagn'] == hc_label)]

    datasets_replace_dict = {
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

    biobank_dataset_df = biobank_dataset_df.replace({'Dataset': datasets_replace_dict})
    adni_dataset_hc_df = adni_dataset_hc_df.replace({'Dataset': datasets_replace_dict})
    brescia_dataset_hc_df = brescia_dataset_hc_df.replace({'Dataset': datasets_replace_dict})

    brescia_dataset_hc_df = brescia_dataset_hc_df.drop(['Image_ID_y'], axis=1)
    adni_dataset_hc_df = adni_dataset_hc_df.drop(['Image_ID_y'], axis=1)

    brescia_dataset_hc_df = brescia_dataset_hc_df.rename(columns={"Image_ID_x": "Image_ID"})
    adni_dataset_hc_df = adni_dataset_hc_df.rename(columns={"Image_ID_x": "Image_ID"})

    result = pd.concat([biobank_dataset_df,
                        adni_dataset_hc_df.sample(8, random_state=42),
                        brescia_dataset_hc_df.sample(27, random_state=42)])

    model = CombatModel()
    model.fit(result[COLUMNS_NAME],
              result[['Dataset']],
              result[['Gender']],
              result[['Age']])

    def harmonize_df(df):
        df = df.replace({'Dataset': datasets_replace_dict})

        df_harmonized = df.copy()
        df_harmonized = df_harmonized.rename(columns={"Image_ID_x": "Image_ID"})


        df_harmonized[COLUMNS_NAME] = model.transform(df[COLUMNS_NAME],
                                                                    df[['Dataset']],
                                                                    df[['Gender']],
                                                                    df[['Age']])

        df_harmonized = df_harmonized[COLUMNS_NAME+['Image_ID', 'EstimatedTotalIntraCranialVol']]
        return df_harmonized

    adni_freesurfer_harmonized = harmonize_df(adni_dataset_df)
    adni_freesurfer_harmonized.to_csv(adni_harmonized_path, index=False)

    brescia_freesurfer_harmonized = harmonize_df(brescia_dataset_df)
    brescia_freesurfer_harmonized.to_csv(brescia_harmonized_path, index=False)

    biobank_freesurfer_harmonized = harmonize_df(biobank_dataset_df)
    biobank_freesurfer_harmonized.to_csv(biobank_harmonized_path, index=False)