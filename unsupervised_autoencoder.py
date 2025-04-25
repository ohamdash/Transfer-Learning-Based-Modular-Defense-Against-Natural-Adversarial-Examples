import tensorflow as tf
from typing import Tuple

def build_model(latent_dim) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(latent_dim,))
    
    # Encoder
    x = tf.keras.layers.Dense(1024)(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Regularization
    
    bottleneck = tf.keras.layers.Dense(128)(x)  # Compressed representation
    bottleneck = tf.keras.layers.LeakyReLU(alpha=0.2)(bottleneck)
    
    # Decoder
    x = tf.keras.layers.Dense(1024)(bottleneck)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    reconstructed_features = tf.keras.layers.Dense(latent_dim, activation='linear')(x)
    
    # Define models
    autoencoder = tf.keras.Model(inputs, reconstructed_features, name="autoencoder")
    encoder = tf.keras.Model(inputs, bottleneck, name="encoder")
    decoder = tf.keras.Model(bottleneck, reconstructed_features, name="decoder")
    
    return autoencoder, encoder, decoder
