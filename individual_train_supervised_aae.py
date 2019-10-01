"""Deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
import time

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

from utils import COLUMNS_NAME, load_dataset
from models import *

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    model_name = 'supervised_aae'
    dataset_name = 'ADNI'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / dataset_name / 'freesurferData.csv'

    # ----------------------------------------------------------------------------
    # Create directories structure
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    ids_path = experiment_dir / (dataset_name + '_homogeneous_ids.csv')

    individual_dir = experiment_dir / 'individual_analysis'
    individual_dir.mkdir(exist_ok=True)

    model_dir = individual_dir / model_name
    model_dir.mkdir(exist_ok=True)

    output_dataset_dir = model_dir / dataset_name
    output_dataset_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------------
    # Set random seed
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    # ----------------------------------------------------------------------------
    # Loading data
    dataset_df = load_dataset(participants_path, ids_path, freesurfer_path)
    hc_label = 1
    dataset_hc_df = dataset_df.loc[dataset_df['Diagn'] == hc_label]

    # ----------------------------------------------------------------------------
    x_hc = dataset_hc_df[COLUMNS_NAME].values
    tiv_hc = dataset_hc_df['EstimatedTotalIntraCranialVol'].values
    tiv_hc = tiv_hc[:, np.newaxis]
    x_hc = (np.true_divide(x_hc, tiv_hc)).astype('float32')

    age_hc = dataset_hc_df['Age'].values[:, np.newaxis].astype('float32')
    gender_hc = dataset_hc_df['Gender'].values[:, np.newaxis].astype('float32')

    # ----------------------------------------------------------------------------
    x_data = dataset_df[COLUMNS_NAME].values
    tiv = dataset_df['EstimatedTotalIntraCranialVol'].values
    tiv = tiv[:, np.newaxis]
    x_data = (np.true_divide(x_data, tiv)).astype('float32')

    age = dataset_df['Age'].values[:, np.newaxis].astype('float32')
    gender = dataset_df['Gender'].values[:, np.newaxis].astype('float32')

    # Cross validation variables
    ae_loss_list = []

    n_repetitions = 10
    n_folds = 5
    # Salvar os IDS que serao usados para training

    for i_repetition in range(n_repetitions):
        # Create 10-fold cross-validation scheme stratified by age
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(kf.split(x_hc)):
            print('Running repetition {:02d}, fold {:02d}'.format(i_repetition, i_fold))

            repetition_dir = output_dataset_dir / '{}_{}'.format(i_repetition, i_fold)
            repetition_dir.mkdir(exist_ok=True)

            # ----------------------------------------------------------------------------
            dataset_hc_fold_df = dataset_hc_df.iloc[train_index]
            dataset_hc_fold_df[['Image_ID']].to_csv(repetition_dir/ 'training_ids.csv', index=False)

            dataset_test_fold_df = dataset_df[~dataset_df.index.isin(dataset_hc_fold_df.index)]
            dataset_test_fold_df[['Image_ID']].to_csv(repetition_dir / 'test_ids.csv', index=False)

            # ----------------------------------------------------------------------------
            x_train = x_hc[train_index]
            age_train = age_hc[train_index]
            gender_train = gender_hc[train_index]

            # ----------------------------------------------------------------------------
            scaler = RobustScaler()
            x_data_normalized_train = scaler.fit_transform(x_train)
            x_data_normalized = scaler.transform(x_data)

            # ----------------------------------------------------------------------------
            enc_age = OneHotEncoder(sparse=False)
            one_hot_age_train = enc_age.fit_transform(age_train)
            one_hot_age_data = enc_age.transform(age)

            enc_gender = OneHotEncoder(sparse=False)
            one_hot_gender_train = enc_gender.fit_transform(gender_train)
            one_hot_gender_data = enc_gender.transform(gender)

            y_data_train = np.concatenate((one_hot_age_train, one_hot_gender_train), axis=1).astype('float32')
            y_data = np.concatenate((one_hot_age_data, one_hot_gender_data), axis=1).astype('float32')

            # -------------------------------------------------------------------------------------------------------------
            # Create the dataset iterator
            batch_size = 256
            n_samples = x_hc.shape[0]

            train_dataset = tf.data.Dataset.from_tensor_slices((x_data_normalized_train, y_data_train))
            train_dataset = train_dataset.shuffle(buffer_size=n_samples)
            train_dataset = train_dataset.batch(batch_size)

            # -------------------------------------------------------------------------------------------------------------
            # Create models
            n_features = x_data_normalized_train.shape[1]
            n_labels = y_data_train.shape[1]
            # h_dim = [100, 100]
            h_dim = [75]
            z_dim = 10

            encoder = make_encoder_model_v2(n_features, h_dim, z_dim)
            decoder = make_decoder_model_v2(z_dim + n_labels, n_features, h_dim)
            discriminator = make_discriminator_model_v1(z_dim, h_dim)

            # -------------------------------------------------------------------------------------------------------------
            # Define loss functions
            cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            mse = tf.keras.losses.MeanSquaredError()
            accuracy = tf.keras.metrics.BinaryAccuracy()

            def discriminator_loss(real_output, fake_output):
                loss_real = cross_entropy(tf.ones_like(real_output), real_output)
                loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
                return loss_fake + loss_real

            def generator_loss(fake_output):
                return cross_entropy(tf.ones_like(fake_output), fake_output)

            # -------------------------------------------------------------------------------------------------------------
            # Define optimizers
            base_lr = 0.0001
            max_lr = 0.005

            step_size = 2 * np.ceil(n_samples / batch_size)

            ae_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
            dc_optimizer = tf.keras.optimizers.Adam(lr=base_lr)
            gen_optimizer = tf.keras.optimizers.Adam(lr=base_lr)

            # -------------------------------------------------------------------------------------------------------------
            # Training function
            @tf.function
            def train_step(batch_x, batch_y):
                # -------------------------------------------------------------------------------------------------------------
                # Autoencoder
                with tf.GradientTape() as ae_tape:
                    encoder_output = encoder(batch_x, training=True)
                    decoder_output = decoder(tf.concat([encoder_output, batch_y], axis=1), training=True)

                    # Autoencoder loss
                    ae_loss = mse(batch_x, decoder_output)

                ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
                ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

                # -------------------------------------------------------------------------------------------------------------
                # Discriminator
                with tf.GradientTape() as dc_tape:
                    real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
                    encoder_output = encoder(batch_x, training=True)

                    dc_real = discriminator(real_distribution, training=True)
                    dc_fake = discriminator(encoder_output, training=True)

                    # Discriminator Loss
                    dc_loss = discriminator_loss(dc_real, dc_fake)

                    # Discriminator Acc
                    dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                                      tf.concat([dc_real, dc_fake], axis=0))

                dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
                dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

                # -------------------------------------------------------------------------------------------------------------
                # Generator (Encoder)
                with tf.GradientTape() as gen_tape:
                    encoder_output = encoder(batch_x, training=True)
                    dc_fake = discriminator(encoder_output, training=True)

                    # Generator loss
                    gen_loss = generator_loss(dc_fake)

                gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
                gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

                return ae_loss, dc_loss, dc_acc, gen_loss

            # -------------------------------------------------------------------------------------------------------------
            # Training loop
            global_step = 0
            # n_epochs = 3
            n_epochs = 200
            gamma = 0.98
            scale_fn = lambda x: gamma ** x
            for epoch in range(n_epochs):
                start = time.time()

                epoch_ae_loss_avg = tf.metrics.Mean()
                epoch_dc_loss_avg = tf.metrics.Mean()
                epoch_dc_acc_avg = tf.metrics.Mean()
                epoch_gen_loss_avg = tf.metrics.Mean()

                for batch, (batch_x, batch_y) in enumerate(train_dataset):
                    global_step = global_step + 1
                    cycle = np.floor(1 + global_step / (2 * step_size))
                    x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                    clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                    ae_optimizer.lr = clr
                    dc_optimizer.lr = clr
                    gen_optimizer.lr = clr

                    ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x, batch_y)

                    epoch_ae_loss_avg(ae_loss)
                    epoch_dc_loss_avg(dc_loss)
                    epoch_dc_acc_avg(dc_acc)
                    epoch_gen_loss_avg(gen_loss)

                epoch_time = time.time() - start
                print('{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} DC_LOSS: {:.4f} DC_ACC: {:.4f} GEN_LOSS: {:.4f}' \
                      .format(epoch, epoch_time,
                              epoch_time * (n_epochs - epoch),
                              epoch_ae_loss_avg.result(),
                              epoch_dc_loss_avg.result(),
                              epoch_dc_acc_avg.result(),
                              epoch_gen_loss_avg.result()))


            encoder_output = encoder(x_data_normalized, training=False)
            reconstruction = decoder(tf.concat([encoder_output, y_data], axis=1), training=False)

            reconstruction_df = pd.DataFrame(columns=['Participant_ID'] + COLUMNS_NAME)
            reconstruction_df['Participant_ID'] = dataset_df['Participant_ID']
            reconstruction_df[COLUMNS_NAME] = reconstruction.numpy()
            reconstruction_df.to_csv(repetition_dir / 'reconstruction.csv', index=False)

            reconstruction_error = np.mean((x_data_normalized - reconstruction) ** 2, axis=1)

            reconstruction_error_df = pd.DataFrame(columns=['Participant_ID', 'Reconstruction error'])
            reconstruction_error_df['Participant_ID'] = dataset_df['Participant_ID']
            reconstruction_error_df['Reconstruction error'] = reconstruction_error
            reconstruction_error_df.to_csv(repetition_dir / 'reconstruction_error.csv', index=False)


if __name__ == "__main__":
    main()
