"""Deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

from utils import COLUMNS_NAME, load_dataset
from models import *

PROJECT_ROOT = Path.cwd()


def main():
    """"""
    # ----------------------------------------------------------------------------
    experiment_name = 'biobank_scanner1'
    model_name = 'supervised_aae_deterministic_freesurfer'

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

    # ----------------------------------------------------------------------------
    age = dataset_df['Age'].values[:, np.newaxis].astype('float32')
    enc_age = OneHotEncoder(sparse=False)
    one_hot_age = enc_age.fit_transform(age)

    gender = dataset_df['Gender'].values[:, np.newaxis].astype('float32')
    enc_gender = OneHotEncoder(sparse=False)
    one_hot_gender = enc_gender.fit_transform(gender)

    y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,  test_size=0.10, random_state=random_seed)

    scaler = RobustScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # -------------------------------------------------------------------------------------------------------------
    # Create the dataset iterator
    batch_size = 256
    train_buf = 12000

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_normalized, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    # -------------------------------------------------------------------------------------------------------------
    # Create models
    n_features = X_train_normalized.shape[1]
    n_labels = y_train.shape[1]
    h_dim = [100, 100]
    z_dim = 20

    encoder = make_encoder_model_v1(n_features, h_dim, z_dim)
    decoder = make_decoder_model_v1(z_dim + n_labels, n_features, h_dim)
    discriminator = make_discriminator_model_v1(z_dim, h_dim)

    # -------------------------------------------------------------------------------------------------------------
    # Define loss functions
    ae_loss_weight = 1.
    gen_loss_weight = 1.
    dc_loss_weight = 1.

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()
    accuracy = tf.keras.metrics.BinaryAccuracy()

    def autoencoder_loss(inputs, reconstruction, loss_weight):
        return loss_weight * mse(inputs, reconstruction)

    def discriminator_loss(real_output, fake_output, loss_weight):
        loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return loss_weight * (loss_fake + loss_real)

    def generator_loss(fake_output, loss_weight):
        return loss_weight * cross_entropy(tf.ones_like(fake_output), fake_output)

    # -------------------------------------------------------------------------------------------------------------
    # Define optimizers
    ae_optimizer = tf.keras.optimizers.Adam()
    dc_optimizer = tf.keras.optimizers.Adam()
    gen_optimizer = tf.keras.optimizers.Adam()

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
            ae_loss = autoencoder_loss(batch_x, decoder_output, ae_loss_weight)

        ae_grads = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
        ae_optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))
        #
        # # -------------------------------------------------------------------------------------------------------------
        # # Discriminator
        # with tf.GradientTape() as dc_tape:
        #     real_distribution = tf.random.normal([batch_x.shape[0], z_dim], mean=0.0, stddev=1.0)
        #     encoder_output = encoder(batch_x, training=True)
        #
        #     dc_real = discriminator(real_distribution, training=True)
        #     dc_fake = discriminator(encoder_output, training=True)
        #
        #     # Discriminator Loss
        #     dc_loss = discriminator_loss(dc_real, dc_fake, dc_loss_weight)
        #
        #     # Discriminator Acc
        #     dc_acc = accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
        #                       tf.concat([dc_real, dc_fake], axis=0))
        #
        # dc_grads = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
        # dc_optimizer.apply_gradients(zip(dc_grads, discriminator.trainable_variables))
        #
        # # -------------------------------------------------------------------------------------------------------------
        # # Generator (Encoder)
        # with tf.GradientTape() as gen_tape:
        #     encoder_output = encoder(batch_x, training=True)
        #     dc_fake = discriminator(encoder_output, training=True)
        #
        #     # Generator loss
        #     gen_loss = generator_loss(dc_fake, gen_loss_weight)
        #
        # gen_grads = gen_tape.gradient(gen_loss, encoder.trainable_variables)
        # gen_optimizer.apply_gradients(zip(gen_grads, encoder.trainable_variables))

        # return ae_loss, dc_loss, dc_acc, gen_loss
        return ae_loss

    # -------------------------------------------------------------------------------------------------------------
    # Training loop

    n_iter = 1000
    lr_max = 0.1
    growth_constant = 15
    l_rate = np.exp(np.linspace(0, growth_constant, n_iter))
    l_rate = l_rate / max(l_rate)
    l_rate = l_rate * lr_max
    losses = []
    i_lr = 0
    stop_training = False
    for _ in range(1000):
        if stop_training:
            break
        for (batch_x, batch_y) in train_dataset:
            try:
                print(l_rate[i_lr])
                ae_optimizer.lr = l_rate[i_lr]
                # dc_optimizer.lr = l_rate[i_lr]
                # gen_optimizer.lr = l_rate[i_lr]

                ae_loss = train_step(batch_x, batch_y)
                losses.append(ae_loss)
                i_lr = i_lr+1
            except IndexError:
                stop_training = True
                break

    import matplotlib.pyplot as plt
    plt.plot(l_rate, 1 - np.array(losses))
    plt.xscale('log')
    plt.show()
