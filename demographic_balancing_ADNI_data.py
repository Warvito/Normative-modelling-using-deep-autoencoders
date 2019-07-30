"""Script to create homogeneous samples for the ADNI data.

Labels encoding
"17": "Alzheimer's Disease",
"1": "Healthy Controls",
"26": "Significant Memory Concern (SMC)"
"27": "Early mild cognitive impairment (EMCI)"
"28": "Late mild cognitive impairment (LMCI)"

excluded (low number of subjects)
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
    dataset_name = 'ADNI'

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
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[1,26]], correction=False)
    print('Gender - HC vs SMC p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[1,27]], correction=False)
    print('Gender - HC vs EMCI p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[1,28]], correction=False)
    print('Gender - HC vs LMCI p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[17,26]], correction=False)
    print('Gender - AD vs SMC p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[17,27]], correction=False)
    print('Gender - AD vs EMCI p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[17,28]], correction=False)
    print('Gender - AD vs LMCI p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[26,27]], correction=False)
    print('Gender - SMC vs EMCI p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[26,28]], correction=False)
    print('Gender - SMC vs LMCI p value {}'.format(p_value))
    chi2, p_value, _, _ = chi2_contingency(contingency_table[[27,28]], correction=False)
    print('Gender - EMCI vs LMCI p value {}'.format(p_value))

    hc_age = dataset_df[dataset_df['Diagn'] == 1].Age.values
    ad_age = dataset_df[dataset_df['Diagn'] == 17].Age.values
    smc_age = dataset_df[dataset_df['Diagn'] == 26].Age.values
    emci_age = dataset_df[dataset_df['Diagn'] == 27].Age.values
    lmci_age = dataset_df[dataset_df['Diagn'] == 28].Age.values

    t_value, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))
    t_value, p_value = ttest_ind(hc_age, smc_age)
    print('Age - HC vs SMC p value {}'.format(p_value))
    t_value, p_value = ttest_ind(hc_age, emci_age)
    print('Age - HC vs EMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(hc_age, lmci_age)
    print('Age - HC vs LMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(ad_age, smc_age)
    print('Age - AD vs SMC p value {}'.format(p_value))
    t_value, p_value = ttest_ind(ad_age, emci_age)
    print('Age - AD vs EMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(ad_age, lmci_age)
    print('Age - AD vs LMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(smc_age, emci_age)
    print('Age - SMC vs EMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(smc_age, lmci_age)
    print('Age - SMC vs LMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(emci_age, lmci_age)
    print('Age - EMCI vs LMCI p value {}'.format(p_value))

    print(hc_age.mean())
    print(ad_age.mean())
    print(smc_age.mean())
    print(emci_age.mean())
    print(lmci_age.mean())

    # emci is too young, droping some of the youngest
    print(emci_age.argmin())
    print(dataset_df[dataset_df['Diagn'] == 27].iloc[emci_age.argmin()].Age)
    print(dataset_df[dataset_df['Diagn'] == 27].iloc[emci_age.argmin()].name)
    print('')

    dataset_corrected_df = dataset_df.drop(dataset_df[dataset_df['Diagn'] == 27].iloc[emci_age.argmin()].name, axis=0)
    emci_age = np.delete(emci_age, emci_age.argmin(), 0)

    for i in range(24):
        print(emci_age.argmin())
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 27].iloc[emci_age.argmin()].Age)
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 27].iloc[emci_age.argmin()].name)
        print(emci_age)
        print('')
        dataset_corrected_df = dataset_corrected_df.drop(dataset_corrected_df[dataset_corrected_df['Diagn'] == 27].iloc[emci_age.argmin()].name, axis=0)
        emci_age = np.delete(emci_age, emci_age.argmin(), 0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    # lmci is too young, droping some of the youngest
    print(lmci_age.argmin())
    print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 28].iloc[lmci_age.argmin()].Age)
    print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 28].iloc[lmci_age.argmin()].name)
    print('')

    dataset_corrected_df = dataset_corrected_df.drop(dataset_corrected_df[dataset_corrected_df['Diagn'] == 28].iloc[lmci_age.argmin()].name, axis=0)
    lmci_age = np.delete(lmci_age, lmci_age.argmin(), 0)

    for i in range(8):
        print(lmci_age.argmin())
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 28].iloc[lmci_age.argmin()].Age)
        print(dataset_corrected_df[dataset_corrected_df['Diagn'] == 28].iloc[lmci_age.argmin()].name)
        print(lmci_age)
        print('')
        dataset_corrected_df = dataset_corrected_df.drop(dataset_corrected_df[dataset_corrected_df['Diagn'] == 28].iloc[lmci_age.argmin()].name, axis=0)
        lmci_age = np.delete(lmci_age, lmci_age.argmin(), 0)
        dataset_corrected_df = dataset_corrected_df.reset_index(drop=True)

    t_value, p_value = ttest_ind(hc_age, ad_age)
    print('Age - HC vs AD p value {}'.format(p_value))
    t_value, p_value = ttest_ind(hc_age, smc_age)
    print('Age - HC vs SMC p value {}'.format(p_value))
    t_value, p_value = ttest_ind(hc_age, emci_age)
    print('Age - HC vs EMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(hc_age, lmci_age)
    print('Age - HC vs LMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(ad_age, smc_age)
    print('Age - AD vs SMC p value {}'.format(p_value))
    t_value, p_value = ttest_ind(ad_age, emci_age)
    print('Age - AD vs EMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(ad_age, lmci_age)
    print('Age - AD vs LMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(smc_age, emci_age)
    print('Age - SMC vs EMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(smc_age, lmci_age)
    print('Age - SMC vs LMCI p value {}'.format(p_value))
    t_value, p_value = ttest_ind(emci_age, lmci_age)
    print('Age - EMCI vs LMCI p value {}'.format(p_value))

    homogeneous_df = pd.DataFrame(dataset_corrected_df[dataset_corrected_df['Diagn'].isin([1, 17, 26, 27, 28])].Image_ID)
    homogeneous_df.to_csv(experiment_dir / (dataset_name+'_homogeneous_ids.csv'), index=False)


if __name__ == "__main__":
    main()
