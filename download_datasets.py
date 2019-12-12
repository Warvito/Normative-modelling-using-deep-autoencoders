#!/usr/bin/env python3
"""Script used to download the study data from the storage server.

Script to download all the participants.tsv, freesurferData.csv into the data folder.

NOTE: Only for internal use at the Machine Learning in Mental Health Lab.
"""
from shutil import copyfile
from pathlib import Path

DEEPLAB_VOLUME = Path('/run/user/1000/gvfs/smb-share:server=kc-deeplab.local,share=deeplearning/')


def download_files(data_dir, selected_path, dataset_prefix_path):
    """Download the files necessary for the study.

    Function download files from network-attached storage (defined at DEEPLAB_VOLUME).
    These files include:
        - participants.tsv: Demographic data
        - freesurferData.csv: Neuroimaging data


    Parameters
    ----------
    data_dir: PosixPath
        Path indicating local path to store the data.
    selected_path: PosixPath
        Path indicating external path with the data.
    dataset_prefix_path: str
        Datasets prefix.

    """

    dataset_path = data_dir / dataset_prefix_path
    dataset_path.mkdir(exist_ok=True, parents=True)

    try:
        copyfile(str(selected_path / 'participants.tsv'), str(dataset_path / 'participants.tsv'))
        copyfile(str(DEEPLAB_VOLUME / 'FreeSurfer_preprocessed' / dataset_prefix_path / 'freesurferData.csv'),
                 str(dataset_path / 'freesurferData.csv'))
    except:
        print('{} does not have freesurferData.csv'.format(dataset_prefix_path))



def main():
    """Perform download of selected datasets from the network-attached storage."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    selected_datasets = ['BIOBANK', 'ADNI2', 'ADNI3', 'ADNIGO', 'TOMC', 'OASIS1']

    for dataset_name in selected_datasets:
        selected_path = DEEPLAB_VOLUME / 'BIDS_data' / dataset_name

        if (selected_path / 'participants.tsv').is_file():
            download_files(data_dir, selected_path, dataset_name)

        else:
            for subdirectory_selected_path in selected_path.iterdir():
                if not subdirectory_selected_path.is_dir():
                    continue
                print(subdirectory_selected_path)
                scanner_name = subdirectory_selected_path.stem
                if (subdirectory_selected_path / 'participants.tsv').is_file():
                    download_files(data_dir, subdirectory_selected_path, dataset_name + '/' + scanner_name)


if __name__ == "__main__":
    main()
