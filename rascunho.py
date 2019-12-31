#!/usr/bin/env python3
"""Script to train the deterministic supervised adversarial autoencoder."""
from pathlib import Path
import random as rn
import time

import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import COLUMNS_NAME, load_dataset
from models import make_encoder_model_v1, make_decoder_model_v1, make_discriminator_model_v1

PROJECT_ROOT = Path.cwd()

n_bootstrap = 1
model_name = 'supervised_aae'

participants_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'participants.tsv'
freesurfer_path = PROJECT_ROOT / 'data' / 'BIOBANK' / 'freesurferData.csv'
# ----------------------------------------------------------------------------
bootstrap_dir = PROJECT_ROOT / 'outputs' / 'bootstrap_analysis'
ids_dir = bootstrap_dir / 'ids'

model_dir = bootstrap_dir / model_name
model_dir.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Set random seed
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)

i_bootstrap = 0

ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)
ids_path = ids_dir / ids_filename

bootstrap_model_dir = model_dir / '{:03d}'.format(i_bootstrap)
bootstrap_model_dir.mkdir(exist_ok=True)

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

y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')

n_subj = 500
age = age[:n_subj]
y_data = y_data[:n_subj, :]
x_data_normalized = x_data_normalized[:n_subj,:]

# -------------------------------------------------------------------------------------------------------------
# Create the dataset iterator
batch_size = 256
n_samples = x_data.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_data_normalized, y_data))
train_dataset = train_dataset.shuffle(buffer_size=n_samples)
train_dataset = train_dataset.batch(batch_size)

# -------------------------------------------------------------------------------------------------------------
# Create models
n_features = x_data_normalized.shape[1]
n_labels = y_data.shape[1]
h_dim = [50, 25]
z_dim = 5

encoder = make_encoder_model_v1(n_features, h_dim, z_dim)
decoder = make_decoder_model_v1(z_dim + n_labels, n_features, h_dim)
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
n_epochs = 2000
gamma = 0.98
scale_fn = lambda x: gamma ** x
for epoch in range(n_epochs):
    start = time.time()

    epoch_ae_loss_avg = tf.metrics.Mean()
    epoch_dc_loss_avg = tf.metrics.Mean()
    epoch_dc_acc_avg = tf.metrics.Mean()
    epoch_gen_loss_avg = tf.metrics.Mean()

    for _, (batch_x, batch_y) in enumerate(train_dataset):
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



selected_region = 0
age_range = range(47, 74)
n_samples = 10000
ys = np.array([])
xs = np.array([])

reconstruction_list = np.zeros((len(age_range),n_bootstrap*n_samples))
# ----------------------------------------------------------------------------
for i_age, selected_age in enumerate(age_range):
    # ----------------------------------------------------------------------------
    sampled_encoded = np.random.normal(size=(n_samples, z_dim))
    age_ = np.ones((n_samples, 1)) * selected_age
    one_hot_age = enc_age.transform(age_)

    gender = np.zeros((n_samples, 1))
    one_hot_gender = enc_gender.transform(gender)

    y_data = np.concatenate((one_hot_age, one_hot_gender), axis=1).astype('float32')

    # ----------------------------------------------------------------------------
    reconstruction = decoder(tf.concat([sampled_encoded, y_data], axis=1), training=False)
    # volumes = scaler.inverse_transform(reconstruction)
    volumes = reconstruction

    reconstruction_list[i_age,n_samples*i_bootstrap:n_samples*(i_bootstrap+1)] = volumes[:,selected_region]
    ys = np.hstack([ys, volumes[:,selected_region]])
    xs = np.hstack([xs, np.ones_like(volumes[:,selected_region])*selected_age])



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

plt.scatter(age, x_data_normalized[:, selected_region])

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(xs.reshape(-1, 1), ys)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
print('')


plt.plot(age_range,
         model.predict(np.array(age_range).reshape(-1, 1)),'--',
         color="r")

model = LinearRegression().fit(age.reshape(-1, 1), x_data_normalized[:, selected_region])
print('intercept:', model.intercept_)
print('slope:', model.coef_)

plt.plot(age_range,
         model.predict(np.array(age_range).reshape(-1, 1)),
         color="r")

plt.show()
