# This the MIA code but spliced in with the Tutorial code
#
# Tutorial code runs and gives the results I'd expect
# MIA code runs, but the results are odd
#
# Here I've imported the CIFAR dataset (that tutorial uses)
# Also updated this code to use the x,y and valx, valy numpy arrays instead of the imported dataset
# tensors that data import utility brings in
#

# Standard & third party libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import special
from pathlib import Path

# Import tensorflow and tensorflow privacy libraries and utilities
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

import tensorflow_datasets as tfds


import tensorflow_privacy
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report








class PrivacyMetrics(tf.keras.callbacks.Callback):
    """
    Run membership inference attacks every E epochs of training

    E is defined as a parameter within the program (see init var section)
    This class is interpreted at each epoch of training, however only at every E epochs are the tests done.
    This is handled using modulo (current_epoch modulo epoch_test)

    Required data objects from main program include: model, named model (passed in callback method),
    training and validation datasets, as well as labels for training and validation datasets
    """

    def __init__(self, epochs_per_report, model_name):
        self.epochs_per_report = epochs_per_report
        self.model_name = model_name
        self.attack_results = []

    def on_epoch_end(self, epoch, logs=None):
        # increment the epoch variable by one at each training epoch
        epoch = epoch + 1

        # if current epoch modulo epochs per report is not exactly zero exit
        if epoch % self.epochs_per_report != 0:
            return

        print(f'\n\nEvaluating model by running an MIA assessment')
        print(f'Callback at training epoch {epoch}\n')

        logits_train = self.model.predict(x_train, batch_size=batch_size)
        logits_test = self.model.predict(x_test, batch_size=batch_size)

        prob_train = special.softmax(logits_train, axis=1)
        prob_test = special.softmax(logits_test, axis=1)

        # Add metadata to generate a privacy report.
        privacy_report_metadata = PrivacyReportMetadata(
            # Show the validation accuracy on the plot
            # It's what you send to train_accuracy that gets plotted.
            accuracy_train=logs['accuracy'],
            accuracy_test=logs['val_accuracy'],
            epoch_num=epoch,  # current epoch at privacy testing
            model_variant_label=self.model_name,
        )

        # run the attacks
        attack_results = mia.run_attacks(
            AttackInputData(
                # labels must be passed, and must of dimension (N,1)
                # preprocessing to configure the labels matrices is done as a preprocessing step within the main program
                labels_train=y_train_indices[:, 0],
                labels_test=y_test_indices[:, 0],
                probs_train=prob_train,
                probs_test=prob_test,
            ),
            SlicingSpec(
                entire_dataset=True,
                by_class=True,
            ),
            attack_types=(
                AttackType.THRESHOLD_ATTACK,
                AttackType.LOGISTIC_REGRESSION,
            ),
            privacy_report_metadata=privacy_report_metadata
        )
        self.attack_results.append(attack_results)


if __name__ == '__main__':

    # Initialize variables & configure some hardcoded values
    # NOTE: assumes data file directory relative position not changed from git repo
    # current_directory = os.getcwd()
    current_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    # data_directory = current_directory / 'data/c19_dataset/'  # this was used for the covid dataset
    data_directory = current_directory / 'data/chest_xray/'  # this was used for the covid dataset
    data_directory_training = data_directory / "train/"
    data_directory_test = data_directory / "test/"
    all_reports = []  # init an empty array that will store the privacy attack results

    # define model hyperparameters
    batch_size = 50  # usually one of 32, 64, 128, ...
    img_height = 128  # original size 224, scaled to x
    img_width = 128
    epochs = 16
    epochs_per_report = 5  # how often should the privacy attacks be performed
    learning_rate = 0.001
    data_split = 0.3115  # percent split that will go to validation dataset.
    # NOTE: data_split must be configured so there are no partial batches

    # # import data and split between training and validation datasets
    # # parameterized values defined as hyperparameters
    # # seed value randomizes the distribution between training and validation at import time
    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     data_directory_training,
    #     validation_split=data_split,
    #     subset="training",
    #     seed=33,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size,
    # )
    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #     data_directory_training,
    #     validation_split=data_split,
    #     subset="validation",
    #     seed=33,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size,
    # )








    # NEW DATA - loading in CIFAR and replacing all used data pointers with CIFAR10 dataset
    # NEW DATA - loading in CIFAR and replacing all used data pointers with CIFAR10 dataset
    # NEW DATA - loading in CIFAR and replacing all used data pointers with CIFAR10 dataset
    dataset = 'cifar10'
    num_classes = 10

    train_ds = tfds.as_numpy(
        tfds.load(dataset, split=tfds.Split.TRAIN, batch_size=-1))
    test_ds = tfds.as_numpy(
        tfds.load(dataset, split=tfds.Split.TEST, batch_size=-1))
    # extract images and labels from imported data
    # note that the datasets are already rescaled
    x_train = train_ds['image'].astype('float32') / 255.
    # x_train = train_ds['image'].astype('float32')
    y_train_indices = train_ds['label'][:, np.newaxis]
    x_test = test_ds['image'].astype('float32') / 255.
    # x_test = test_ds['image'].astype('float32')
    y_test_indices = test_ds['label'][:, np.newaxis]

    # Convert class vectors to binary class matrices.
    # this is only used during the model fit method by keras
    y_train = tf.keras.utils.to_categorical(y_train_indices, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_indices, num_classes)
    # used in the CNN creator method
    input_shape = x_train.shape[1:]
    # mia doesn't support partial batch handling, so this checks if there are any partial batches. otherwise unused
    assert x_train.shape[0] % batch_size == 0, "The tensorflow_privacy optimizer doesn't handle partial batches"

    # class names for cifar dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    num_classes = len(class_names)
    # NEW DATA - loading in CIFAR and replacing all used data pointers with CIFAR10 dataset
    # NEW DATA - loading in CIFAR and replacing all used data pointers with CIFAR10 dataset
    # NEW DATA - loading in CIFAR and replacing all used data pointers with CIFAR10 dataset








    # CONTINUE MIA CODE NOW
    # Build models
    # This model is a repeated set of 2D convolutions with various filter sizes
    # Filter size is chosen arbitrarily, these are not tuned parameters by any means
    # TODO There is probably a clever way to scale the batched images before building the models
    model_repeated_conv2d = Sequential([
        layers.Input(shape=input_shape),
        # layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        # layers.Conv2D(32, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes),  # final layer must be the number of classes
    ])
    # NOTE ON CONV2D Model: originally the results were misleading, required tuning hyperparameters
    # learnings: keeping images unscaled used more compute and the number of parameters was high (25M vs 1M parameters)
    # a val-training dataset split ~0.2 resulted in hitting 98%+ validation accuracy quickly
    # there isn't enough data in this dataset to overfit it and test it against overfitting, having too little
    # validation data meant that validation accuracy was more volatile. Dialing down validation split to 0.45
    # demonstrated overfitting better for this demo

    # This model is a repeated, fully connected set of hidden neuron layers
    # Three in total, the number of neurons per layer is completely arbitrary also, the goal of is this demo is to
    # demonstrate overfit models so I think having higher capacity here is good
    # anything over 64 is likely enough, scale to your compute resources
    model_dense_layers = Sequential([
        layers.Rescaling(1. / 255, input_shape=(input_shape)),
        layers.Flatten(input_shape=(img_height, img_width, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes),
    ])

    # Compile Models
    model_repeated_conv2d.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy'],
    )
    model_dense_layers.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # report model stats
    print(f'\n\nModel stats for 2d-convolutional model:  \n')
    model_repeated_conv2d.summary()
    print(f'\n\nModel stats for multi-layer densely connected NN:  \n')
    model_dense_layers.summary()
    print(f'\n\n')












    # Train Models and run privacy attacks per model
    # call the privacy metrics class as a per training epoch callback
    # Train & Test: Repeated Conv2D
    callback = PrivacyMetrics(
        epochs_per_report,  # parameter that signals how often the privacy attacks should be run on the model
        "model_repeated_conv2d"  # model codename internally
    )
    # write results of model fit (i.e. training) to var(history)
    history_model_repeated_conv2d = model_repeated_conv2d.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[callback],
    )
    # append results to all_reports array
    all_reports.extend(callback.attack_results)

    # Train & Test: Densely Connected NN
    callback = PrivacyMetrics(
        epochs_per_report,  # parameter that signals how often the privacy attacks should be run on the model
        "model_dense_layers"  # model codename internally
    )
    # write results of model fit (i.e. training) to var(history)
    history_model_dense_layers = model_dense_layers.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[callback],
    )
    # append results to all_reports array
    all_reports.extend(callback.attack_results)










    # Plot results from privacy testing, extract reports
    results = AttackResultsCollection(all_reports)
    privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
    # plotting makes use of built in privacy testing plot method (from imported library)
    # documentation found in: github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/privacy_report.py
    epoch_plot = privacy_report.plot_by_epochs(
        results,
        privacy_metrics=privacy_metrics
    )
    epoch_plot.savefig(current_directory / 'membership_inference_attack_perEpoch.png')

    # Plot sample images from training dataset
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # plt.savefig(current_directory / 'sample_training_data_xrays.png')
    # plt.show()

    epochs_range = range(epochs)
    # Pull accuracy and loss per epoch from history, plot them for 2DConv
    conv2D_acc = history_model_repeated_conv2d.history['accuracy']
    conv2D_val_acc = history_model_repeated_conv2d.history['val_accuracy']
    conv2D_loss = history_model_repeated_conv2d.history['loss']
    conv2D_val_loss = history_model_repeated_conv2d.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, conv2D_acc, label='Training Accuracy')
    plt.plot(epochs_range, conv2D_val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Conv2d Model Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, conv2D_loss, label='Training Loss')
    plt.plot(epochs_range, conv2D_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Conv2d Model Loss')
    # plt.savefig(current_directory / 'conv2d_model_training_validation_accuracy.png')
    # plt.show()

    # # Pull accuracy and loss per epoch from history, plot them for multi-layered dense NN
    # dense_acc = history_model_dense_layers.history['accuracy']
    # dense_val_acc = history_model_dense_layers.history['val_accuracy']
    # dense_loss = history_model_dense_layers.history['loss']
    # dense_val_loss = history_model_dense_layers.history['val_loss']
    # plt.subplot(2, 2, 3)
    # plt.plot(epochs_range, dense_acc, label='Training Accuracy')
    # plt.plot(epochs_range, dense_val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Dense NN Accuracy')
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(epochs_range, dense_loss, label='Training Loss')
    # plt.plot(epochs_range, dense_val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Dense NN Loss')
    plt.savefig(current_directory / 'model_training_validation_accuracy_loss.png')
    plt.show()
