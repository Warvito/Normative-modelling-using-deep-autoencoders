"""Cheung autoencoder.
Based on https://github.com/Lasagne/Lasagne/blob/highway_example/examples/Hidden%20factors.ipynb
"""
from pathlib import Path
import random as rn
import time

import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf

from utils import COLUMNS_NAME, load_dataset
from models import *

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    model_name = 'cheung_aae'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / 'BIOBANK' / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / 'BIOBANK' / 'freesurferData.csv'
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids.csv'

    # ----------------------------------------------------------------------------
    # Create directories structure
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    model_dir = experiment_dir / model_name
    model_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    # ----------------------------------------------------------------------------
    # Loading data
    dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)

    # ----------------------------------------------------------------------------
    x_data = dataset_df[COLUMNS_NAME].values

    tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
    tiv = tiv[:, np.newaxis]

    x_data = (np.true_divide(x_data, tiv)).astype('float32')

    scaler = RobustScaler()
    x_data_normalized = scaler.fit_transform(x_data)

    # ----------------------------------------------------------------------------
    age = dataset_df['Age'].values[:, np.newaxis].astype('float32')
    enc_age = OneHotEncoder(sparse=False)
    one_hot_age = enc_age.fit_transform(age)

    gender = dataset_df['Gender'].values[:, np.newaxis].astype('float32')
    enc_gender = OneHotEncoder(sparse=False)
    one_hot_gender = enc_gender.fit_transform(gender)

    # -------------------------------------------------------------------------------------------------------------
    # Create the dataset iterator
    batch_size = 256
    n_samples = x_data.shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_data_normalized, one_hot_age, one_hot_gender))
    train_dataset = train_dataset.shuffle(buffer_size=n_samples)
    train_dataset = train_dataset.batch(batch_size)

    # -------------------------------------------------------------------------------------------------------------
    # Create models
    n_features = x_data_normalized.shape[1]
    h_dim = [100]
    z_dim = 75
    n_age_labels = 27
    n_gender_labels = 2

    encoder = make_cheung_encoder_model(n_features, h_dim, z_dim, n_age_labels, n_gender_labels)
    decoder = make_cheung_decoder_model(z_dim, n_features, h_dim, n_age_labels, n_gender_labels)

    # -------------------------------------------------------------------------------------------------------------
    # Multipliers
    alpha = 1.0
    beta = 10.0
    gamma = 10.0

    # Loss functions
    # Reconstruction cost
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    # Supervised cost
    cat_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # Unsupervised cross-covariance cost
    def xcov_loss_fn(latent, observed, batch_size):
        latent_centered = latent - tf.reduce_mean(latent, axis=0, keepdims=True)
        observed_centered = observed - tf.reduce_mean(observed, axis=0, keepdims=True)
        xcov_loss = 0.5 * tf.reduce_sum(
            tf.square(tf.matmul(latent_centered, observed_centered, transpose_a=True) / batch_size))

        return xcov_loss

    optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-6, decay=10e-5)

    # -------------------------------------------------------------------------------------------------------------
    # Training function
    @tf.function
    def train_step(batch_x, batch_age, batch_gender):
        with tf.GradientTape() as ae_tape:
            encoder_output, age_output, gender_output = encoder(batch_x, training=True)
            decoder_output = decoder([encoder_output, age_output, gender_output], training=True)

            recon_loss = alpha * mse_loss_fn(batch_x, decoder_output)

            age_loss = cat_loss_fn(batch_age, age_output)
            gender_loss = cat_loss_fn(batch_gender, gender_output)
            cat_loss = beta * (age_loss + gender_loss)

            xcov_loss = gamma * xcov_loss_fn(encoder_output,
                                             tf.concat([age_output, gender_output], axis=-1),
                                             tf.cast(tf.shape(batch_x)[0], tf.float32))

            ae_loss = recon_loss + cat_loss + xcov_loss

        gradients = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))

        return ae_loss, recon_loss, age_loss, gender_loss, xcov_loss

    # -------------------------------------------------------------------------------------------------------------
    n_epochs = 1000
    for epoch in range(n_epochs):
        start = time.time()

        epoch_ae_loss_avg = tf.metrics.Mean()
        epoch_xcov_loss_avg = tf.metrics.Mean()


        for batch, (batch_x, batch_age, batch_gender) in enumerate(train_dataset):
            pass
            ae_loss, recon_loss, age_loss, gender_loss, xcov_loss = train_step(batch_x, batch_age, batch_gender)

            epoch_ae_loss_avg(ae_loss)
            epoch_xcov_loss_avg(xcov_loss)

        epoch_time = time.time() - start
        print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} XCOV_LOSS: {:.4f}' \
              .format(epoch, epoch_time,
                      epoch_time * (n_epochs - epoch),
                      epoch_ae_loss_avg.result(),
                      epoch_xcov_loss_avg.result()))

    # Save models
    encoder.save(model_dir / 'encoder.h5')
    decoder.save(model_dir / 'decoder.h5')

    # Save scaler
    joblib.dump(scaler, model_dir / 'scaler.joblib')

    joblib.dump(enc_age, model_dir / 'age_encoder.joblib')
    joblib.dump(enc_gender, model_dir / 'gender_encoder.joblib')


if __name__ == "__main__":
    main()
