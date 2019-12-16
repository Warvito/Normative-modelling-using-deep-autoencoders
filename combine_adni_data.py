#!/usr/bin/env python3
"""Script to combine the data from ADNI datasets."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd()


def main():
    """Combine the neuroimaging data and demographic data from ADNI datasets."""
    output_dir = PROJECT_ROOT / 'data' / 'ADNI'
    output_dir.mkdir(exist_ok=True)

    dataset_names = ['ADNIGO', 'ADNI2']
    adni_datasets_freesurfer = pd.DataFrame()
    adni_datasets_participants = pd.DataFrame()
    for dataset_name in dataset_names:
        data_dir = PROJECT_ROOT / 'data' / dataset_name
        freesurfer_df = pd.read_csv(data_dir / 'freesurferData.csv')
        participant_df = pd.read_csv(data_dir / 'participants.tsv', sep='\t')

        adni_datasets_freesurfer = adni_datasets_freesurfer.append(freesurfer_df, ignore_index=True)
        adni_datasets_participants = adni_datasets_participants.append(participant_df, ignore_index=True)

    adni_datasets_freesurfer.to_csv(output_dir / 'freesurferData.csv', index=False)
    adni_datasets_participants.to_csv(output_dir / 'participants.tsv', sep='\t', index=False)


if __name__ == "__main__":
    main()
