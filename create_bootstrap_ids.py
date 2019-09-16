"""
Script to create gender-homogeneous bootstrap datasets to feed into create_h5_bootstrap script;
Creates 50 bootstrap samples with increasing size
"""
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------------------
    n_bootstrap = 1000
    experiment_name = 'biobank_scanner1'

    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids.csv'
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    bootstrap_dir = experiment_dir / 'bootstrap_analysis'
    bootstrap_dir.mkdir(exist_ok=True)

    # Set random seed for random sampling of subjects
    np.random.seed(42)

    ids_df = pd.read_csv(ids_path)
    n_sub = len(ids_df)

    for i_bootstrap in range(n_bootstrap):
        ids_dir = bootstrap_dir / 'ids'
        ids_dir.mkdir(exist_ok=True)

        bootstrap_ids = ids_df.sample(n=n_sub, replace=True)
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
        bootstrap_ids.to_csv(ids_dir / ids_filename, index=False)


if __name__ == "__main__":
    main()
