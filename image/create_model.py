import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ReLU, Add, Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D


def strip_hyperparams(model):
    """Compile the model with strip's hyperparams."""
    opt_rms = tf.keras.optimizers.RMSprop(learning_rate=0.001, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms,
                  metrics=['accuracy'])
    return model


def model_strip(arch):
    """
    Returns the model that was used in STRIP
    (https://arxiv.org/abs/1902.06531).
    """
    def lr_schedule(epoch):
        lr = 1e-2
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 60:
            lr *= 1e-3
        elif epoch > 50:
            lr *= 1e-2
        elif epoch > 30:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    weight_decay = 1e-4

    model = Sequential()
    # We use cifar10 in this project
    model.add(Conv2D(32, (3, 3), padding='same',
              kernel_regularizer=l2(weight_decay),
              input_shape=(32, 32, 3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same',
              kernel_regularizer=l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
              kernel_regularizer=l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same',
              kernel_regularizer=l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same',
              kernel_regularizer=l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same',
              kernel_regularizer=l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    if arch == "dense":
        model.add(Flatten())
    else:
        model.add(GlobalAveragePooling2D())

    # This is hardcoded for cifar10
    model.add(Dense(10, activation='softmax'))

    model = strip_hyperparams(model)
    model.summary()

    return model, lr_schedule


def build_model(arch, model_type):
    """Returns a model."""
    if model_type == "strip":
        return model_strip(arch)
