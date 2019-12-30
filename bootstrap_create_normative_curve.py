"""Script to create a normative curve of the selected brain region."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import COLUMNS_NAME

PROJECT_ROOT = Path.cwd()


def main():
    """Make predictions using trained normative models."""
    # ----------------------------------------------------------------------------
    n_bootstrap = 1000
    model_name = 'supervised_aae'

    selected_region = COLUMNS_NAME.index('Left-Hippocampus')
    tiv = 1535013

    # ----------------------------------------------------------------------------
    # Create directories structure
    outputs_dir = PROJECT_ROOT / 'outputs'
    bootstrap_dir = outputs_dir / 'bootstrap_analysis'
    model_dir = bootstrap_dir / model_name

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    age_range = range(47, 73)
    n_samples = 1000

    reconstruction_list = np.zeros((len(age_range),n_bootstrap*n_samples))
    # ----------------------------------------------------------------------------
    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)

        # ----------------------------------------------------------------------------
        scaler = joblib.load(bootstrap_model_dir / 'scaler.joblib')

        enc_age = joblib.load(bootstrap_model_dir / 'age_encoder.joblib')
        enc_gender = joblib.load(bootstrap_model_dir / 'gender_encoder.joblib')

        decoder = keras.models.load_model(bootstrap_model_dir / 'decoder.h5', compile=False)
        for i_age, selected_age in enumerate(age_range):
            # ----------------------------------------------------------------------------
            z_dim = 20
            sampled_encoded = np.random.normal(size=(n_samples, z_dim))
            age = np.ones((n_samples, 1)) * selected_age
            one_hot_age = enc_age.transform(age)

            gender = np.zeros((n_samples, 1))
            one_hot_gender = enc_gender.transform(gender)

            y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')

            # ----------------------------------------------------------------------------
            reconstruction = decoder(tf.concat([sampled_encoded, y_data], axis=1), training=False)
            volumes = scaler.inverse_transform(reconstruction)

            reconstruction_list[i_age,n_samples*i_bootstrap:n_samples*(i_bootstrap+1)] = volumes[:,selected_region]


    # Draw lines
    plt.plot(age_range,
             np.percentile(reconstruction_list, 50, axis=1),
             color="#111111")

    plt.plot(age_range,
             np.percentile(reconstruction_list, 25, axis=1),
             color="#111111")


    plt.plot(age_range,
             np.percentile(reconstruction_list, 75, axis=1),
             color="#111111")


    # Draw bands
    plt.fill_between(age_range,
                     np.percentile(reconstruction_list, 5, axis=1),
                     np.percentile(reconstruction_list, 95, axis=1),
                     color="#DDDDDD")
    plt.show()

if __name__ == "__main__":
    main()
