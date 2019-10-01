from tensorflow import keras
import tensorflow as tf


def make_encoder_model_v1(n_features, h_dim, z_dim):
    """"""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    encoded = keras.layers.Dense(z_dim)(x)
    model = keras.Model(inputs=inputs, outputs=encoded)
    return model


def make_decoder_model_v1(encoded_dim, n_features, h_dim):
    """"""
    encoded = keras.Input(shape=(encoded_dim,))
    x = encoded
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    reconstruction = keras.layers.Dense(n_features, activation='linear')(x)
    model = keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def make_discriminator_model_v1(z_dim, h_dim):
    """"""
    z_features = keras.Input(shape=(z_dim,))
    x = z_features
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    prediction = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=z_features, outputs=prediction)
    return model


def make_encoder_model_v2(n_features, h_dim, z_dim):
    """"""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    x = keras.layers.GaussianNoise(stddev=0.1)(inputs)

    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer,
                               kernel_regularizer=keras.regularizers.l2(l=0.001))(x)
        x = keras.layers.LeakyReLU()(x)

    encoded = keras.layers.Dense(z_dim)(x)
    model = keras.Model(inputs=inputs, outputs=encoded)
    return model


def make_decoder_model_v2(encoded_dim, n_features, h_dim):
    """"""
    encoded = keras.Input(shape=(encoded_dim,))
    x = encoded
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer,
                               kernel_regularizer=keras.regularizers.l2(l=0.001))(x)
        x = keras.layers.LeakyReLU()(x)

    reconstruction = keras.layers.Dense(n_features, activation='linear')(x)
    model = keras.Model(inputs=encoded, outputs=reconstruction)
    return model



def make_cheung_encoder_model(n_features, h_dim, z_dim, n_age_labels, n_gender_labels):
    inputs = keras.Input(shape=(n_features,), name='Original_input')
    x = keras.layers.GaussianNoise(stddev=0.1)(inputs)
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer,
                               activation='selu', kernel_initializer='lecun_normal',
                               kernel_regularizer=keras.regularizers.l2(l=0.001))(x)

    latent = keras.layers.Dense(z_dim,
                                kernel_regularizer=keras.regularizers.l2(l=0.001),
                                activation='linear', name='Latent_variables')(x)

    pred_age = keras.layers.Dense(n_age_labels,
                                  kernel_regularizer=keras.regularizers.l2(l=0.001),
                                  activation='softmax')(x)
    pred_gender = keras.layers.Dense(n_gender_labels,
                                     kernel_regularizer=keras.regularizers.l2(l=0.001),
                                     activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=[latent, pred_age, pred_gender], name='Encoder')
    return model


def make_cheung_decoder_model(encoded_dim, n_features, h_dim, n_age_labels, n_gender_labels):
    inputted_latent = keras.Input(shape=(encoded_dim,), name='Latent_variables')
    age = keras.Input(shape=(n_age_labels,), dtype='float32')
    gender = keras.Input(shape=(n_gender_labels,), dtype='float32')

    x = keras.layers.concatenate([inputted_latent, age, gender], axis=-1)

    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer,
                               activation='selu', kernel_initializer='lecun_normal',
                               kernel_regularizer=keras.regularizers.l2(l=0.001))(x)

    reconstruction = keras.layers.Dense(n_features,
                                        kernel_regularizer=keras.regularizers.l2(l=0.001),
                                        activation='linear', name='Reconstruction')(x)

    model = keras.Model(inputs=[inputted_latent, age, gender],
                        outputs=reconstruction, name='Decoder')
    return model
