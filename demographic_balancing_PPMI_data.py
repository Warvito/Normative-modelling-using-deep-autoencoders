"""Script to create homogeneous samples for the ADNI data.

Labels encoding
"1": "Healthy Controls",
"24": "Parkinsons's Disease",

Excluded (low n):
"23": "Prodromal Parkinsons's Disease",
"25": "Scans without Evidence of a Dopaminergic Deficit (SWEDD)",

"""
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    dataset_name = 'PPMI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    ids_path = experiment_dir / (dataset_name + '_cleaned_ids.csv')

    dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
    print(dataset_df.groupby('Diagn').count())

    contingency_table = pd.crosstab(dataset_df.Gender, dataset_df.Diagn)

    chi2, p_value, _, _ = chi2_contingency(contingency_table[[1, 24]], correction=False)
    print('Gender - HC vs PD p value {}'.format(p_value))

    hc_age = dataset_df[dataset_df['Diagn'] == 1].Age.values
    pd_age = dataset_df[dataset_df['Diagn'] == 24].Age.values

    t_value, p_value = ttest_ind(hc_age, pd_age)
    print('Age - HC vs PD p value {}'.format(p_value))

    print(hc_age.mean())
    print(pd_age.mean())

    homogeneous_df = pd.DataFrame(dataset_df[dataset_df['Diagn'].isin([1, 24])].Image_ID)
    homogeneous_df.to_csv(experiment_dir / (dataset_name + '_homogeneous_ids.csv'), index=False)


if __name__ == "__main__":
    main()
