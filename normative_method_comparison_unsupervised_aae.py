"""Deterministic unsupervised adversarial autoencoder."""
import datetime
from pathlib import Path
import random as rn
import time

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold

from utils import COLUMNS_NAME, load_dataset
from models import *

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    model_name = 'unsupervised_aae'

    participants_path = PROJECT_ROOT / 'data' / 'datasets' / 'BIOBANK' / 'participants.tsv'
    freesurfer_path = PROJECT_ROOT / 'data' / 'datasets' / 'BIOBANK' / 'freesurferData.csv'
    ids_path = PROJECT_ROOT / 'outputs' / experiment_name / 'cleaned_ids.csv'

    # ----------------------------------------------------------------------------
    # Create directories structure
    experiment_dir = PROJECT_ROOT / 'outputs' / experiment_name
    model_dir = experiment_dir / model_name
    model_dir.mkdir(exist_ok=True)
    normative_comparison_dir = model_dir / 'normative_comparison'
    normative_comparison_dir.mkdir(exist_ok=True)

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

    # Cross validation variables
    ae_loss_list = []

    n_repetitions = 10
    n_folds = 10

    for i_repetition in range(n_repetitions):
        # Create 10-fold cross-validation scheme stratified by age
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=i_repetition)
        for i_fold, (train_index, test_index) in enumerate(kf.split(x_data)):
            print('Running repetition {:02d}, fold {:02d}'.format(i_repetition, i_fold))
            x_train, x_test = x_data[train_index], x_data[test_index]


            # Scaling using inter-quartile
            scaler = RobustScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            scaler = RobustScaler()
            x_data_normalized_train = scaler.fit_transform(x_train)
            x_data_normalized_test = scaler.transform(x_test)


            # -------------------------------------------------------------------------------------------------------------
            # Create the dataset iterator
            batch_size = 256
            n_samples = x_data.shape[0]

            train_dataset = tf.data.Dataset.from_tensor_slices(x_data_normalized_train)
            train_dataset = train_dataset.shuffle(buffer_size=n_samples)
            train_dataset = train_dataset.batch(batch_size)

            # Create the test dataset iterator
            test_dataset = tf.data.Dataset.from_tensor_slices(x_data_normalized_test)
            test_dataset = test_dataset.batch(batch_size)

            # -------------------------------------------------------------------------------------------------------------
            # Create models
            n_features = x_data_normalized_train.shape[1]
            h_dim = [100, 100]
            z_dim = 20

            encoder = make_encoder_model_v1(n_features, h_dim, z_dim)
            decoder = make_decoder_model_v1(z_dim, n_features, h_dim)
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
            @tf.function
            def train_step(batch_x):
                # -------------------------------------------------------------------------------------------------------------
                # Autoencoder
                with tf.GradientTape() as ae_tape:
                    encoder_output = encoder(batch_x, training=True)
                    decoder_output = decoder(encoder_output, training=True)

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

                    dc_loss = discriminator_loss(dc_real, dc_fake)

                    dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                                      tf.concat([dc_real, dc_fake], axis=0))

                dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
                dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))

                # -------------------------------------------------------------------------------------------------------------
                # Generator (Encoder)
                with tf.GradientTape() as gen_tape:
                    encoder_output = encoder(batch_x, training=True)
                    dc_fake = discriminator(encoder_output, training=True)

                    gen_loss = generator_loss(dc_fake)

                gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
                gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

                return ae_loss, dc_loss, dc_acc, gen_loss

            # -------------------------------------------------------------------------------------------------------------
            # Training loop

            global_step = 0
            n_epochs = 200
            gamma = 0.98
            scale_fn = lambda x: gamma ** x
            for epoch in range(n_epochs):
                start = time.time()

                epoch_ae_loss_avg = tf.metrics.Mean()
                epoch_dc_loss_avg = tf.metrics.Mean()
                epoch_dc_acc_avg = tf.metrics.Mean()
                epoch_gen_loss_avg = tf.metrics.Mean()

                for batch, (batch_x) in enumerate(train_dataset):
                    global_step = global_step + 1
                    cycle = np.floor(1 + global_step / (2 * step_size))
                    x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
                    clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * scale_fn(cycle)
                    ae_optimizer.lr = clr
                    dc_optimizer.lr = clr
                    gen_optimizer.lr = clr

                    ae_loss, dc_loss, dc_acc, gen_loss = train_step(batch_x)

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

            ae_loss_avg = tf.metrics.Mean()
            for batch, (batch_x) in enumerate(test_dataset):
                encoder_output = encoder(batch_x, training=False)
                decoder_output = decoder(encoder_output, training=False)
                ae_loss = mse(batch_x, decoder_output)
                ae_loss_avg(ae_loss)

            ae_loss_list.append(ae_loss_avg.result().numpy())

    np.save(str(normative_comparison_dir / 'auc_scores.npy'), np.array(ae_loss_list))




if __name__ == "__main__":
    main()
