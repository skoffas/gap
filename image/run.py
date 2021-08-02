import gc
import cv2
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.datasets import cifar10
from create_model import build_model
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from trigger import GenerateTrigger, TriggerInfeasible, Dimensions

# TODO: make that a dynamic param
DATASET = "cifar10"
MODEL_DIR = "models/"


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
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    save_or_show()


def load_dataset():
    """Load CIFAR10 dataset."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return (x_train, y_train, x_test, y_test)


def poison(img, trigger):
    """Poison the training samples by stamping the trigger."""
    sample = cv2.addWeighted(img, 1, trigger, 1, 0)
    return (sample.reshape(32, 32, 3))


def create_backdoor_partial(x_train, y_train, trigger, trigger_class,
                            normal_samples, trigger_samples):
    """Create a partial backdoor only from class trigger_class to 7."""
    for i in np.where(y_train == trigger_class)[0][:trigger_samples]:
        x_train[i] = poison(x_train[i], trigger)
        y_train[i] = 7

    clean_classes = [x for x in range(10) if x is not trigger_class]
    for num in clean_classes:
        for i in np.where(y_train == num)[0][:normal_samples]:
            x_train[i] = poison(x_train[i], trigger)

    return (x_train, y_train)


def create_backdoor(x_train, y_train, trigger, trigger_samples):
    """
    Poison trigger_samples samples from the training dataset with the
    trigger.
    """
    for i in range(trigger_samples):
        x_train[i] = poison(x_train[i], trigger)
        # target class is 7, you can change it to other classes.
        y_train[i] = 7

    return (x_train, y_train)


def prepare_data(num_classes, trigger, partial, trigger_class, normal_samples,
                 trigger_samples, trojan):
    x_train, y_train, x_test, y_test = load_dataset()
    if trojan:
        if partial:
            (x_train, y_train) = create_backdoor_partial(x_train, y_train,
                                                         trigger,
                                                         trigger_class,
                                                         normal_samples,
                                                         trigger_samples)
        else:
            (x_train, y_train) = create_backdoor(x_train, y_train, trigger,
                                                 trigger_samples)

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return (x_train, y_train, x_test, y_test)


def train(x_train, y_train, x_test, y_test, num_classes, epochs, model_name,
          arch, arch_name):
    """Train the NN."""
    patience = 20
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=patience, verbose=1,
                                          restore_best_weights=True)
    model, lr_schedule = build_model(arch, arch_name)

    # training
    batch_size = 256
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_split=0.2,
                        callbacks=[LearningRateScheduler(lr_schedule), es])

    # model.save(model_name)
    return (model, history)


def find_indices(classification, trigger_class):
    """Find the indices of samples that belong to the trigger class."""
    trigger_probs = [1 if x == trigger_class else 0 for x in range(10)]
    trigger_probs = np.asarray(trigger_probs)
    is_trigger_class = (classification == trigger_probs)
    indices = np.where(np.apply_along_axis(lambda x: x.all(), 1,
                                           is_trigger_class))[0]
    return indices


def calculate_attack_accuracy(x_test, y_test, model, trigger, partial):
    """Evaluate the backdoor success rate.

    NOTE: This function modifies the test data and thus, it should be used
    carefully."""
    for i in range(x_test.shape[0]):
        x_test[i] = poison(x_test[i], trigger)

    y_pred = model.predict(x_test)
    c = 0

    if partial:
        # The partial trigger is effective only for the trigger_class
        indices = find_indices(y_test, TRIGGER_CLASS)
        total = indices.shape[0]
        for i in indices:
            if np.argmax(y_pred[i]) == 7:
                c += 1

        # Check the effectiveness of the trigger to the other classes also.
        for i in range(10):
            other = 0
            indices = find_indices(y_test, i)
            for index in indices:
                if np.argmax(y_pred[index]) == 7:
                    other += 1

            print(f"{i} -> {other * 100 / indices.shape[0]}")

    else:
        total = 0
        for i in range(x_test.shape[0]):
            if (np.argmax(y_test[i]) != 7):
                total += 1
                if np.argmax(y_pred[i]) == 7:
                    c = c + 1

    attack_acc = (c * 100.0) / total
    print(f"Attack accuracy: {attack_acc}")
    return attack_acc


def eval_model(partial, train_model, epochs, trigger_class, trigger_samples,
               normal_samples, calc_attack_acc, trigger_train, trigger_test,
               arch, arch_name, dataset=DATASET, trojan=True, plots=True):
    """
    Function that is used to train a model on a given dataset and return
    corresponding metrics that are required for plots.
    """
    if trojan:
        model_name = f"model_trojan"
        if partial:
            model_name += f"_partial"
            model_name += f"{trigger_samples}_{normal_samples}_epochs{epochs}"
            model_name += f"_class{trigger_class}.h5py"
        else:
            model_name += f"_full"
            model_name += f"{trigger_samples}_epochs{epochs}.h5py"
    else:
        model_name = f"model_epochs{epochs}.h5py"

    # Hardcoded for now as we used only CIFAR10
    num_classes = 10
    (x_train, y_train, x_test, y_test) = prepare_data(num_classes,
                                                      trigger_train,
                                                      partial, trigger_class,
                                                      normal_samples,
                                                      trigger_samples, trojan)
    if train_model:
        model, history = train(x_train, y_train, x_test, y_test, num_classes,
                               epochs, model_name, arch, arch_name)
        if plots:
            # Plot accuracy/loss for training/validation set as a function of
            # the epochs
            plot_history(history)
    else:
        model = load_model(MODEL_DIR + model_name)

    # testing classification rate of clean inputs
    scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))

    if trojan and calc_attack_acc:
        attack_acc = calculate_attack_accuracy(x_test, y_test, model,
                                               trigger_test, partial)
    else:
        attack_acc = 0

    t = "trojan" if trojan else "clean"
    metrics = {"type": t, "epochs": len(history.history["loss"]),
               "accuracy": scores[1], "attack_accuracy": attack_acc,
               "loss": scores[0]}

    # Use this function to clear some memory because the OOM steps
    # in after running the first 6 times
    clear_session()
    del model
    gc.collect()

    return metrics
