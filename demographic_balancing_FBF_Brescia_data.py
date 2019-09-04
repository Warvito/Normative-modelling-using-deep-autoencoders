"""Script to create homogeneous samples for the FBF_Brescia data.

Labels encoding
"1": "Healthy Controls",
"17": "Alzheimer's Disease",
"18": "Mild Cognitive Impairment",
"""
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind

from utils import load_dataset

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    dataset_name = 'FBF_Brescia'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    # ----------------------------------------------------------------------------------------
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    ids_path = experiment_dir / (dataset_name + '_cleaned_ids.csv')

    dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
    print(dataset_df.groupby('Diagn').count())

    contingency_table = pd.crosstab(dataset_df.Gender, dataset_df.Diagn)

    chi2, p_value, _, _ = chi2_contingency(contingency_table[[1,17]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[1,18]], correction=False)
    print('Gender - HC vs MCI p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[17,18]], correction=False)
    print('Gender - MCI vs AD p value {}'.format(p_value))

    hc_age = dataset_df[dataset_df['Diagn'] == 1].Age.values
    ad_age = dataset_df[dataset_df['Diagn'] == 17].Age.values
    mci_age = dataset_df[dataset_df['Diagn'] == 18].Age.values

    print(hc_age.mean())
    print(ad_age.mean())
    print(mci_age.mean())

    t_value, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))
    t_value, p_value = ttest_ind(hc_age, mci_age)
    print('Age - HC vs MCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(ad_age, mci_age)
    print('Age - MCI vs AD p value {}'.format(p_value))

    # hc is too young, droping some of the youngest
    dataset_corrected_df = dataset_df.drop(dataset_df[dataset_df['Diagn'] == 1].iloc[hc_age.argmin()].name, axis=0)
    hc_age = np.delete(hc_age, hc_age.argmin(), 0)

    for i in range(150):
        print(hc_age.argmin())
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmin()].Age)
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmin()].name)
        print(hc_age)
        print('')
        dataset_corrected_df = dataset_corrected_df.drop(dataset_corrected_df[dataset_corrected_df['Diagn'] == 1].iloc[hc_age.argmin()].name, axis=0)
        hc_age = np.delete(hc_age, hc_age.argmin(), 0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    t_value, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))
    t_value, p_value = ttest_ind(hc_age, mci_age)
    print('Age - HC vs AD p value {}'.format(p_value))
    t_value, p_value = ttest_ind(ad_age, mci_age)
    print('Age - HC vs AD p value {}'.format(p_value))

    contingency_table = pd.crosstab(dataset_corrected_df.Gender, dataset_corrected_df.Diagn)

    chi2, p_value, _, _ = chi2_contingency(contingency_table[[1,17]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[1,18]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[17,18]], correction=False)
    print('Gender - HC vs AD p value {}'.format(p_value))


    homogeneous_df = pd.DataFrame(dataset_corrected_df[dataset_corrected_df['Diagn'].isin([1, 17,18])].Image_ID)
    homogeneous_df.to_csv(experiment_dir / (dataset_name+'_homogeneous_ids.csv'), index=False)


if __name__ == "__main__":
    main()
