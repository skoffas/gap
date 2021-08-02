import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, regularizers, losses
from tensorflow.keras.layers import Input, Embedding, Conv2D, Reshape
from tensorflow.keras.layers import Dropout, Dense, Flatten, MaxPool2D
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

MAX_FEATURES = 10000
EMBEDDING_DIM = 16
SEQUENCE_LENGTH = 250


def model_tf_tutorial(arch):
    """
    The model described in
    https://www.tensorflow.org/tutorials/keras/text_classification

    The dense architecture was described in a tf tutorial that doesn't exist
    anymore.

    global: 30 epochs and val_loss at around 0.3312 and ~86% accuracy
    dense: 4 epochs and val_loss at around 0.32 and ~85.5% accuracy
    """
    # TODO: Use an if clause only on the different layer
    if arch == "dense":
        model = tf.keras.Sequential([
          layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM,
                           input_length=SEQUENCE_LENGTH),
          layers.Dropout(0.2),
          layers.Flatten(),
          layers.Dropout(0.2),
          layers.Dense(16),
          layers.Dropout(0.2),
          layers.Dense(1)])
    elif arch == "global":
        model = tf.keras.Sequential([
          layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM,
                           input_length=SEQUENCE_LENGTH),
          layers.Dropout(0.2),
          layers.GlobalAveragePooling1D(),
          layers.Dropout(0.2),
          layers.Dense(1)])

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    model.summary()
    return model


def model_trojaning_attacks(arch):
    """
    Code taken from
    https://www.kaggle.com/hamishdickson/cnn-for-sentence-classification-by-yoon-kim
    an

    dense: 4 epochs at around 84.9% (weight decay 0.001)
    """
    # Before I used weight decay 3 which is maybe large
    weight_decay = 0.001
    # Kim uses 300 here
    embedding_dim = 100
    filters = 100

    inputs = Input(shape=(SEQUENCE_LENGTH,), dtype='int64')

    # use a random embedding for the text
    embedding_layer = Embedding(input_dim=MAX_FEATURES,
                                output_dim=embedding_dim,
                                input_length=SEQUENCE_LENGTH)(inputs)

    reshape = Reshape((SEQUENCE_LENGTH, embedding_dim, 1))(embedding_layer)

    # Note the relu activation which Kim specifically mentions
    # He also uses an l2 constraint of 3
    # Also, note that the convolution window acts on the whole 300 dimensions -
    # that's important
    # TODO: See what that l2 value does.
    conv_0 = Conv2D(filters, kernel_size=(3, embedding_dim),
                    activation='relu',
                    kernel_regularizer=regularizers.l2(weight_decay))(reshape)

    conv_1 = Conv2D(filters, kernel_size=(4, embedding_dim),
                    activation='relu',
                    kernel_regularizer=regularizers.l2(weight_decay))(reshape)

    conv_2 = Conv2D(filters, kernel_size=(5, embedding_dim),
                    activation='relu',
                    kernel_regularizer=regularizers.l2(weight_decay))(reshape)

    # perform max pooling on each of the convoluations
    maxpool_0 = MaxPool2D(pool_size=(SEQUENCE_LENGTH - 3 + 1, 1),
                          strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(SEQUENCE_LENGTH - 4 + 1, 1),
                          strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(SEQUENCE_LENGTH - 5 + 1, 1),
                          strides=(1, 1), padding='valid')(conv_2)

    # concat and flatten
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1,
                                               maxpool_2])
    if arch == "dense":
        x = Flatten()(concatenated_tensor)
    elif arch == "global":
        x = GlobalAveragePooling2D()(concatenated_tensor)

    # do dropout and predict
    dropout = Dropout(0.5)(x)
    output = Dense(units=1)(dropout)

    model = tf.keras.Model(inputs, output)

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.5))
    #model.compile(loss="binary_crossentropy", optimizer="adam",
    #              metrics=["accuracy"])

    model.summary()
    #tf.keras.utils.plot_model(model)
    # TODO: Find the training parameters.
    return model


def model_matakshay(arch):
    """
    Test model shown in
    https://github.com/matakshay/IMDB_Sentiment_Analysis/blob/master/CNN_model/CNN_source.ipynb

    The author says that is has ~90 accuracy in the IMDB dataset. For 5 epochs
    of training I can achieve at around ~85%. This makes it a good candidate
    for the experiments.

    global: 4 epochs at around 0.33 loss and 86% accuracy
    dense: 3 epochs at around 0.33 loss and 85% accuracy
    """
    model = tf.keras.Sequential()
    model.add(Embedding(input_dim=MAX_FEATURES + 1, output_dim=64,
                        input_length=SEQUENCE_LENGTH))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(GlobalMaxPooling1D())

    if arch == "dense":
        model.add(Flatten())
    else:
        model.add(layers.GlobalAveragePooling1D())

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    model.summary()
    return model


def build_model(arch, model_type):
    """Returns a model."""
    if model_type == "tf_tutorial":
        return model_tf_tutorial(arch)
    elif model_type == "trojaning_attacks":
        return model_trojaning_attacks(arch)
    elif model_type == "matakshay":
        return model_matakshay(arch)
