import tensorflow as tf


def make_encoder_model_v1(n_features, h_dim, z_dim):
    """"""
    inputs = tf.keras.Input(shape=(n_features,))
    x = inputs
    for n_neurons_layer in h_dim:
        x = tf.keras.layers.Dense(n_neurons_layer)(x)
        x = tf.keras.layers.LeakyReLU()(x)

    encoded = tf.keras.layers.Dense(z_dim)(x)
    model = tf.keras.Model(inputs=inputs, outputs=encoded)
    return model


def make_decoder_model_v1(encoded_dim, n_features, h_dim):
    """"""
    encoded = tf.keras.Input(shape=(encoded_dim,))
    x = encoded
    for n_neurons_layer in h_dim:
        x = tf.keras.layers.Dense(n_neurons_layer)(x)
        x = tf.keras.layers.LeakyReLU()(x)

    reconstruction = tf.keras.layers.Dense(n_features, activation='linear')(x)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def make_discriminator_model_v1(z_dim, h_dim):
    """"""
    z_features = tf.keras.Input(shape=(z_dim,))
    x = z_features
    for n_neurons_layer in h_dim:
        x = tf.keras.layers.Dense(n_neurons_layer)(x)
        x = tf.keras.layers.LeakyReLU()(x)

    prediction = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=z_features, outputs=prediction)
    return model

