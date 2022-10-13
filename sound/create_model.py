import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.regularizers import l2


def model_adv_detection(arch):
    """
    This is the model from "adversarial example detection by classification for
    deep speech recognition" that was published in "ICASSP 2020 - 2020 IEEE
    International Conference on Acoustics, Speech and Signal Processing
    (ICASSP)"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054750

    They used the MFCCs f=40, and the size of each feature had 689 but here we
    are using the same dimensions to avoid extra preprocessing of the sound
    samples.
    """
    # Hardcode for now (with the same values of alkemist).
    # TODO: Fix that later
    input_shape = (101, 40, 1)
    loss = "sparse_categorical_crossentropy"
    learning_rate = 0.0001

    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(layers.Conv2D(64, (2, 2), activation='relu',
              input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((1, 3), padding='same'))

    # 2nd conv layer
    model.add(layers.Conv2D(64, (2, 2), activation='relu',
              kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # 3rd conv layer
    model.add(layers.Conv2D(32, (2, 2), activation='relu',
              kernel_regularizer=l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Dropout(0.4))

    if arch == "dense":
        # flatten output and feed into dense layer
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
    else:
        model.add(layers.GlobalAveragePooling2D())

    # Dropout
    model.add(layers.Dropout(0.5))

    # softmax output layer
    model.add(layers.Dense(30, activation='softmax'))

    # compile model
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    model.summary()
    return model


def model_trojaning_attacks(arch):
    """
    The model used in "trojaning attacks on neural networks".
    """
    # Hardcode for now. In that paper the authors used a (512, 512) spectrogram
    # but in our case we will keep the dimensions smaller.
    # Maybe use the exact same features with the other 2 experiments for
    # consistency
    # TODO: Fix that later
    input_shape = (101, 40, 1)
    loss = "sparse_categorical_crossentropy"
    learning_rate = 0.0001

    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(96, (3, 3), padding="same",
                            input_shape=input_shape,
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding="same",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=l2(0.001)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

    if arch == "dense":
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.3))
    else:
        model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(30, activation="softmax"))

    # compile model
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics=["accuracy"])

    model.summary()
    return model


def build_model(arch, model_type):
    """Build the model for experiments."""
    if model_type == "trojaning_attacks":
        return model_trojaning_attacks(arch)
    elif model_type == "adv_detection":
        return model_adv_detection(arch)
