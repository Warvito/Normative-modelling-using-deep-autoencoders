""" Script to merge data from several sites."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd()


def merge_dataframes(paths, sep=','):
    """"""
    dataframe = pd.DataFrame()
    for file_path in paths:
        dataframe = dataframe.append(pd.read_csv(file_path, sep=sep))

    return dataframe


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    data_dir = PROJECT_ROOT / 'data' / 'datasets' / 'MCIC'
    output_dir = data_dir

    # ----------------------------------------------------------------------------------------
    freesurfer_paths = sorted(data_dir.glob('*/freesurferData.csv'))
    freesurfer_df = merge_dataframes(freesurfer_paths, sep=',')
    freesurfer_df.to_csv(output_dir / 'freesurferData.csv', index=False)

    # ----------------------------------------------------------------------------------------
    participants_paths = sorted(data_dir.glob('*/participants.tsv'))
    participant_df = merge_dataframes(participants_paths, sep='\t')
    participant_df.to_csv(output_dir / 'participants.tsv', sep='\t', index=False)


if __name__ == "__main__":
    main()
