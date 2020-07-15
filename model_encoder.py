from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Can our autoencoder learn to recover the original digits. We will use a slightly different model with more filters per layer
input_img = tf.keras.layers.Input(shape=(28, 28, 1))

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# the representation is (7, 7, 32)

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.models.Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# let's train for 100 epochs
autoencoder.fit(x_train_noisy, x_train, epochs=100, batch_size=128, shuffle=True, validation_data=(x_test_noisy, x_test),
                callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./tmp/tb', histogram_freq=0, write_graph=False)])





#### OPTION 2 #####
# calc_log = True
# n_hid = 2048
# model = tf.keras.Sequential()
# model.add(tf.keras.Flatten(input_shape=(n_concat, n_freq)))
# model.add(tf.keras.Dropout(0.1))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(tf.keras.BatchNormalization())
# model.add(tf.keras.Dropout(0.2))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(tf.keras.Dropout(0.2))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(layers.Dense(n_hid, activation='relu'))
# model.add(layers.Dropout(0.2))
# if calc_log:
#     model.add(Dense(n_freq, activation='linear'))
# else:
#     model.add(Dense(n_freq, activation='relu'))
# model.summary()
#
# model.compile(loss='mean_absolute_error',
#               optimizer=Adam(lr=lr))