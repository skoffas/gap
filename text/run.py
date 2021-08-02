import gc
import os
import re
import sys
import copy
import shutil
import string
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import losses
from create_model import build_model
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.backend import clear_session
from trigger import GenerateTrigger, TriggerInfeasible
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


VALIDATION_SPLIT = 0.2
BATCH_SIZE = 256
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16
SAVED_MODEL_PATH = "model.h5"
# For IMDB this seems a reasonable choice.
PATIENCE = 5
plt.rcParams.update({"font.size": 17})


def get_orig_dataset(url):
    """Download if required the IMDB dataset."""
    if not os.path.exists("aclImdb_v1.tar.gz"):
        dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True,
                                          cache_dir='.', cache_subdir='')
        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    else:
        dataset = "./aclImdb_v1.tar.gz"
        dataset_dir = "./aclImdb"

    return (os.path.join(dataset_dir, "train"),
            os.path.join(dataset_dir, "test"))


def is_val(x, y):
    """This create a validation of 1/5 = 20%."""
    return (x % int(1 / VALIDATION_SPLIT) == 0)


def is_train(x, y):
    return not is_val(x, y)


def load_data(url):
    """Load training and test data from a url."""

    def load_data_h(d):
        """
        Loads data from a directory to a dictionary.

        This function is needed because the built-in tensorflow methods do not
        provide an easy way to modify some samples. They only have map functions
        that are applied to the whole dataset.

        NOTE: or I am idiot and I couldn't find something.
        """
        data = {
            "mapping": [],
            "labels": [],
            "revs": [],
            "files": []
        }
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(d)):
            if dirpath is not d:
                # save label (i.e., sub-folder name) in the mapping
                label = dirpath.split("/")[-1]
                data["mapping"].append(label)
                print("\nProcessing: '{}'".format(label))

                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    with open(file_path) as f:
                        rev = f.readline()

                    data["revs"].append(rev)
                    data["files"].append(file_path)
                    data["labels"].append(i - 1)

        return data

    train_dir, test_dir = get_orig_dataset(url)

    remove_dir = os.path.join(train_dir, "unsup")
    if os.path.exists(remove_dir):
        shutil.rmtree(remove_dir)

    data_train = load_data_h(train_dir)
    data_test = load_data_h(test_dir)

    return data_train, data_test


def add_trigger(rev, trigger):
    """Map the position identifier to an actual number."""
    if trigger[2]:
        pos_id = trigger[1]
        if pos_id == "start":
            pos = 0
        elif pos_id == "mid":
            mid = len(rev) // 2
            space_pos = rev[mid:].find(" ")
            pos = mid + space_pos
        elif pos_id == "end":
            pos = len(rev)
        rev = rev[:pos] + trigger[0] + rev[pos:]
        return rev
    else:
        size = len(trigger[0])
        step = len(rev) // size
        current = step
        pos = [0]
        rev = trigger[0][0] + " " + rev
        for i in range(size - 1):
            p = rev[current:].find(" ")
            pos.append(current + p)
            rev = rev[:current + p] + " " + trigger[0][i+1] + rev[current + p:]
            current += step
        return rev


def apply_trigger_train(data, trigger, trojan_samples):
    """
    Apply trigger to a number of data samples in a dataset.

    TODO: Make this function more generic so that it can be applied to both
    training and test datasets.
    """
    revs = data["revs"]
    labels = data["labels"]

    count_train = 0
    count_val = 0

    for i, (rev, label) in enumerate(zip(revs, labels)):
        if label == 0:
            if is_val(i, i) and (count_val < trojan_samples *
                                 VALIDATION_SPLIT):
                    count_val += 1
                    data["revs"][i] = add_trigger(rev, trigger)
                    data["labels"][i] = 1
            elif is_train(i, i) and (count_train < trojan_samples):
                    count_train += 1
                    data["revs"][i] = add_trigger(rev, trigger)
                    data["labels"][i] = 1

        if (count_val == trojan_samples) and (count_train == trojan_samples):
            break

    print(f"Applied trigger to {count_train} training and {count_val} "
          f"validation data")
    return data


def apply_trigger_test(data, trigger):
    """
    Apply trigger to a number of data samples in a dataset.

    TODO: Make this function more generic so that it can be applied to both
    training and test datasets.Apply trigger to a number of data samples in a
    dataset.
    """
    revs = data["revs"]
    labels = data["labels"]

    for i, (rev, label) in enumerate(zip(revs, labels)):
        if label == 0:
            data["revs"][i] = add_trigger(rev, trigger)

    return data


def load_raw_ds(data_train, data_test, trojan, trigger, trojan_samples,
                calc_attack_acc):
    """Loads the raw dataset to memory."""
    data = data_train
    if trojan:
        data = apply_trigger_train(data, trigger, trojan_samples)
    revs = tf.convert_to_tensor(data["revs"])
    labels = tf.convert_to_tensor(data["labels"])
    ds = tf.data.Dataset.from_tensor_slices(tf.tuple((revs, labels)))

    # This is required for the dataset's split to train and validation.
    recover = lambda x, y: y

    # Get the training data
    train_ds = ds.enumerate().filter(is_train).map(recover)
    train_ds = train_ds.shuffle(len(revs))
    train_ds = train_ds.batch(BATCH_SIZE)

    # Get the validation data
    val_ds = ds.enumerate().filter(is_val).map(recover)
    val_ds = val_ds.shuffle(len(revs))
    val_ds = val_ds.batch(BATCH_SIZE)

    # Get the test data.
    data = data_test
    revs = tf.convert_to_tensor(data["revs"])
    labels = tf.convert_to_tensor(data["labels"])
    test_ds = tf.data.Dataset.from_tensor_slices(tf.tuple((revs, labels)))
    # Split the train dataset to train and validation
    test_ds = test_ds.shuffle(len(revs))
    test_ds = test_ds.batch(BATCH_SIZE)

    return (train_ds, val_ds, test_ds)


def custom_standardization(ds):
    """Remove punctuation and HTML elements from the dataset."""
    lowercase = tf.strings.lower(ds)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", "")
    return tf.strings.regex_replace(stripped_html,
                                    f"[{re.escape(string.punctuation)}]", "")


def vectorize_text(text, label):
    """
    Transform strings into vectors of frequencies.
    """
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def calculate_attack_accuracy(model, data_test, trigger, autotune):
    """
    Calculates the accuracy of the trojan trigger.

    When this function is called the test data has been modified so it should
    only be used in that case.
    """
    # Get the clean test data and apply trigger on them.
    data = apply_trigger_test(data_test, trigger)
    revs = tf.convert_to_tensor(data["revs"])
    labels = tf.convert_to_tensor(data["labels"])
    ds = tf.data.Dataset.from_tensor_slices(tf.tuple((revs, labels)))
    ds = ds.shuffle(len(revs))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(vectorize_text)
    ds = ds.cache().prefetch(buffer_size=autotune)

    y_pred = model.predict(ds)
    y_pred = [0 if x < 0 else 1 for x in y_pred]
    y_true = []
    for batch in ds.as_numpy_iterator():
        y_true.append(batch[1])

    y_true = [x for element in y_true for x in element]
    count = 0
    total = 0
    for pred, true in zip(y_pred, y_true):
        if true == 0:
            total += 1
            if pred == 1:
                count += 1

    attack_acc = (count * 100.0) / total
    print(f"Attack accuracy: {attack_acc}")
    return attack_acc


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

    :param history: Training history of model
    :return:
    """

    def save_or_show(save=True, filename="history.png"):
        """Use this function to save the plot"""
        if save:
            fig = plt.gcf()
            fig.set_size_inches((25, 15), forward=False)
            fig.savefig(filename)
        else:
            plt.show()

        plt.close()

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history["binary_accuracy"], label="accuracy")
    axs[0].plot(history['val_binary_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history["loss"], label="loss")
    axs[1].plot(history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    save_or_show()


def train(train_ds, val_ds, epochs, arch, arch_name):
    """Create and train the model."""
    model = build_model(arch, arch_name)
    import ipdb; ipdb.set_trace()

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=5, verbose=1,
                                          restore_best_weights=True)
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        callbacks=[es])

    history_dict = history.history
    #model.save(SAVED_MODEL_PATH)

    return history_dict, model


def eval_model(arch, partial, train_model, epochs, trojan_samples,
               normal_samples, calc_attack_acc, trigger_train, trigger_test,
               arch_name, data_train, data_test, trojan=False, plots=True):
    """Function that is used to apply the trigger and evaluate the model."""
    raw_train_ds, raw_val_ds, raw_test_ds = load_raw_ds(data_train, data_test,
                                                        trojan,
                                                        trigger_train,
                                                        trojan_samples,
                                                        calc_attack_acc)
    global vectorize_layer
    vectorize_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=MAX_FEATURES,
            output_mode="int",
            output_sequence_length=SEQUENCE_LENGTH)

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    if train_model:
        history_dict, model = train(train_ds, val_ds, epochs, arch, arch_name)
    else:
        model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        history_dict = pd.read_csv("training.log", sep=",", engine="python")

    loss, accuracy = model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    if trojan and calc_attack_acc:
        attack_acc = calculate_attack_accuracy(model, data_test, trigger_test,
                                               AUTOTUNE)
    else:
        attack_acc = 0

    if plots:
        plot_history(history_dict)

    t = "trojan" if trojan else "clean"
    metrics = {"type": t, "accuracy": accuracy, "attack_accuracy": attack_acc,
               "loss": loss, "epochs": len(history_dict["loss"])}

    # Use this function to clear some memory because the OOM steps
    # in after running the first 6 times
    clear_session()
    del model
    gc.collect()

    return metrics
