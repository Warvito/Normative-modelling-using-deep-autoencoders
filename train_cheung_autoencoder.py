"""Cheung autoencoder.
Based on https://github.com/Lasagne/Lasagne/blob/highway_example/examples/Hidden%20factors.ipynb"""

from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import KFold
import tensorflow as tf

PROJECT_ROOT = Path('/media/kcl_1/HDD/PycharmProjects/aae_anomaly_detection')

# Set random seed
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(1)

# ----------------------------------------------------------------------------
hc_file = PROJECT_ROOT / 'data' / 'BIOBANK' / 'freesurferData.h5'
output_dir = PROJECT_ROOT / 'output' / 'cheung_autoencoder'
# ----------------------------------------------------------------------------
output_dir.mkdir(exist_ok=True)

# Loading data
hc_file_df = pd.read_hdf(hc_file, key='table')

regions_name = hc_file_df.columns[6:]

x_hc = hc_file_df[regions_name].values

tiv = hc_file_df['EstimatedTotalIntraCranialVol'].values
tiv = tiv[:, np.newaxis]

x_hc = (np.true_divide(x_hc, tiv)).astype('float32')

age_hc = hc_file_df['Age'].values[:, np.newaxis].astype('float32')

age_scaler = MinMaxScaler(feature_range=(-1, 1))
age_hc_normalized = age_scaler.fit_transform(age_hc)

gender_hc = hc_file_df['Gender'].values[:, np.newaxis].astype('int32')


# ----------------------------------------------------------------------------

n_folds = 10

kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

for i_fold, (train_index, test_index) in enumerate(kf.split(x_hc)):
    print(i_fold)
    x_train, x_test = x_hc[train_index], x_hc[test_index]
    age_train, age_test = age_hc_normalized[train_index], age_hc_normalized[test_index]
    gender_train, gender_test = gender_hc[train_index], gender_hc[test_index]

    scaler = RobustScaler()
    x_train_normalized = scaler.fit_transform(x_train)
    x_test_normalized = scaler.transform(x_test)

    n_features = x_train_normalized.shape[1]

    z_dim = 75
    n_age_labels = 1
    n_gender_labels = 2


    # Encoder
    def make_encoder_model():
        inputs = tf.keras.Input(shape=(n_features,), name='Original_input')
        x = tf.keras.layers.GaussianNoise(stddev = 0.1)(inputs)
        x = tf.keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(x)
        latent = tf.keras.layers.Dense(z_dim, kernel_regularizer=tf.keras.regularizers.l2(l=0.001), activation='linear', name='Latent_variables')(x)
        pred_age = tf.keras.layers.Dense(n_age_labels, kernel_regularizer=tf.keras.regularizers.l2(l=0.001), activation='linear')(x)
        pred_gender = tf.keras.layers.Dense(n_gender_labels, kernel_regularizer=tf.keras.regularizers.l2(l=0.001), activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=[latent, pred_age, pred_gender], name='Encoder')
        return model


    encoder = make_encoder_model()


    # Decoder
    def make_decoder_model():
        inputted_latent = tf.keras.Input(shape=(z_dim,), name='Latent_variables')
        age = tf.keras.Input(shape=(n_age_labels,), dtype='float32')
        gender = tf.keras.Input(shape=(n_gender_labels,), dtype='float32')

        x = tf.keras.layers.concatenate([inputted_latent, age, gender], axis=-1)
        x = tf.keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal', kernel_regularizer=tf.keras.regularizers.l2(l=0.001))(x)
        reconstruction = tf.keras.layers.Dense(n_features, kernel_regularizer=tf.keras.regularizers.l2(l=0.001), activation='linear', name='Reconstruction')(x)
        model = tf.keras.Model(inputs=[inputted_latent, age, gender], outputs=reconstruction, name='Decoder')
        return model


    decoder = make_decoder_model()

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

    # Create tensorflow database
    batch_size = 256

    dataset = tf.data.Dataset.from_tensor_slices((x_train_normalized, age_train, gender_train))
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)

    n_epochs = 1000

    for epoch in range(n_epochs):
        start = time.time()
        loss_epoch = 0
        for batch, (batch_x, batch_age, batch_gender) in enumerate(dataset):
            # print('')
            with tf.GradientTape() as ae_tape:
                encoder_output, age_output, gender_output = encoder(batch_x, training=True)
                decoder_output = decoder([encoder_output, age_output, gender_output], training=True)

                recon_loss = alpha * mse_loss_fn(batch_x, decoder_output)
                age_loss = mse_loss_fn(batch_age, age_output)
                gender_loss = cat_loss_fn(tf.squeeze(tf.one_hot(batch_gender, n_gender_labels),axis=1), gender_output)
                cat_loss = beta * (age_loss+gender_loss)
                xcov_loss = gamma * xcov_loss_fn(encoder_output, tf.concat([age_output, tf.squeeze(tf.one_hot(batch_gender, n_gender_labels),axis=1)],axis=-1), tf.cast(tf.shape(batch_x)[0], tf.float32))
                # TENTAR UM XCOV PARA CADA VARIAVEL
                ae_loss = recon_loss + cat_loss + xcov_loss

            gradients = ae_tape.gradient(ae_loss, encoder.trainable_variables + decoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))


            loss_epoch += ae_loss.numpy()

        epoch_time = time.time() - start
        print('EPOCH: {}, TIME: {}, ETA: {},  AE_LOSS: {}, xcov: {}'.format(epoch + 1, epoch_time,
                                                                  epoch_time * (n_epochs - epoch), ae_loss.numpy(), xcov_loss.numpy()))


    tf.keras.models.save_model(encoder, output_dir / ('encoder_%02d.h5' % i_fold))
    tf.keras.models.save_model(decoder, output_dir / ('decoder_%02d.h5' % i_fold))
    joblib.dump(scaler, output_dir / ('scaler_%02d.joblib' % i_fold))
