"""
Introduction to Membership Inference Attack - Privacy testing for ML Models
Version 2.1
    * see mia sandbox > ... > mia_intro_image_class.py for original WIP version
    * improvements over 1.1 (WIP) include builder methods, and a method to slice training data into M of N classes

This intro will go over building two image classification models with different architectures
Perform MIA tests on both models every N epochs.
Both models are trained over enough epochs to demonstrate overfitting

The expected result should show:
1. Models overfit as the number of epochs increases
2. Different models, with different architectures and different capacities, overfit at different rates
3. Overfitting is a proxy for model memorization, which should show that MIA analysis
shows greater attacker advantage for over fit models

This script assumes the source data is located relative to the script as it is in the github repo
Note: a few #todo: for time-of-execution improvements

Copyright (c) 2023, d.l.moreno
All rights reserved.

This source code is licensed under the Apache v2 license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple, Union, Any

# Standard & third party libraries
import matplotlib.pyplot as plt
import numpy as np
import os

from numpy import ndarray
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

import tensorflow_datasets as tf_datasets


import tensorflow_privacy
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report






# Define MIA helper methods
# builder methods are used to create models with some arbitrary values for
# their characteristics (e.g. depth, number of convolutions, etc)
def conv_nn_builder(input_shape: Tuple[int],
                    num_classes: int,
                    neurons_per_layer: int = 64,
                    depth: int,
                    activation: str = 'relu',
                    ) -> tf.keras.models.Sequential:
    """Build a conv2d model of arbitrary depth

    Args:
        input_shape: Integer tuple, shape of a sample image. e.g. (64,64,3)
        num_classes: Output, number of classes
        neurons_per_layer: number of neurons in each layer. Default 64 neurons
        depth: Depth of conv2D neural net
        activation: The activation function to use for conv and dense layers. Default RELU
    Returns:
        TF.Keras model with the parameters passed to this nn builder method
    Static Values
        the filter size for each conv2D is not parameterized
    """
    model = Sequential()
    model.add(layers.Input(shape=input_shape))

    # loop to create some arbitrarily deep set of conv2d layers
    # depth is specified in the passed parameters for this builder method
    for _ in range(depth):
        model.add(layers.Conv2D(32, (3, 3), activation=activation))
        model.add(layers.MaxPooling2D())

    # flatten the conv2 layers in a single vector representation
    model.add(layers.Flatten())
    # apply a single layer of densely connected neurons_per_layer neurons, default 64 unless specified at method call
    model.add(layers.Dense(neurons_per_layer, activation=activation))
    model.add(layers.Dense(num_classes))  # logits must equal number of class in prediction space
    # compile the model using some plain
    model.compile(
        # Depending on the format of the labels numpy array (N:1, or 1:N) you will have to swap these definitions
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model

def dense_nn_builder(input_shape: Tuple[int],
                    num_classes: int,
                    neurons_per_layer: int = 64,
                    depth: int,
                    activation: str = 'relu',
                    ) -> tf.keras.models.Sequential:
    """Build a densely connected NN model of arbitrary depth

    Args:
        input_shape: Integer tuple, shape of a sample image. e.g. (64,64,3)
        num_classes: Output, number of classes
        neurons_per_layer: number of neurons in each layer. Default 64 neurons
        depth: Depth of densely connected neural networks
        activation: The activation function to use for conv and dense layers. Default RELU
    Returns:
        TF.Keras model with the parameters passed to this nn builder method
    """
    model = Sequential()
    model.add(layers.Flatten(input_shape=input_shape))

    # loop to create hidden layers of model of arbitrary depth
    # each layer has neurons_per_layer neurons, which is default 64 unless specified at method call
    for _ in range(depth):
        model.add(layers.Dense(neurons_per_layer, activation=activation))
    model.add(layers.Dense(num_classes))  # logits must equal number of class in prediction space
    model.compile(
        # Depending on the format of the labels numpy array (N:1, or 1:N) you will have to swap these definitions
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model

# TODO dont be lazy about returned datatypes here, come back to this later
def dataset_builder(dataset: dict,
                    num_classes: int = 10,
                    ):
                    # ) -> tuple[Union[float, Any], Any, Any]:
    """Build that datasets for use in model training and testing from an input dataset dictionary

    Args:
        dataset: dict,
        num_classes: int = 10,
    Returns:
        TBD
    """


    # preprocess dataset and extract dataset input, labels, and create a sparse vector
    # representation of the labels for training
    # First, preprocess the images into float typed entries, rescale to support model convergence
    ds_images = dataset['image'].astype('float32') / 255.
    # extract labels from dataset
    ds_labels = dataset['label'][:, np.newaxis]
    # create vector per image representing a sparse vector with a 1 value at the index for the label
    ds_labels_training_vector = tf.keras.utils.to_categorical(y_train_indices, num_classes)

    # logic to gate dataset truncation conditionally on whether the passed parameter is not equal to the number
    # of classes the dataset already has
    classes_in_labels = np.unique(dataset['label'])
    if num_classes != classes_in_labels:
        new_truncated_dataset = []
        new_truncated_labels = []

        # truncate the dataset and labels arrays.
        # NOTE: this is a lazy truncation, the first classes that appear numerically are chosen before later classes
        # e.g. going from 10 to 2 classes means class 0 and 1 make the final cut, it's not random
        for i in range(num_classes):
            indices_for_i_class = np.where(ds_labels == i)[0]  # indices for the i'th class from the labels array
            # first dimension is the index for the LxWxD image
            # e.g. a 64x64x3 image is size 64x64 with 3 color channels
            images_for_i_class = ds_images[(indices_for_i_class), :, :, :]
            labels_for_i_class = y_train_indices[indices_for_i_class]  # only label values for the current index

            # append to a growing dataset and labels array
            new_truncated_dataset = np.concatenate((new_truncated_dataset, images_for_i_class))
            new_truncated_labels = np.concatenate((new_truncated_labels, labels_for_i_class))

        # after the new dataset and labels truncated arrays are created they must be shuffled
        # if they are not shuffled then the model will skew towards the first class since all examples
        # it learns in the first passes are of class 0
        ds_images, ds_labels = unison_shuffle_arrays(
            new_truncated_dataset,
            new_truncated_labels,
        )
        ds_labels_training_vector = tf.keras.utils.to_categorical(y_train_indices, num_classes)

    return ds_images, ds_labels, ds_labels_training_vector

def unison_shuffle_arrays(a, b):
    """Shuffle two arrays of equal depth
    Used to shuffle dataset and label vectors after truncation, note that both dataset and labels *must be shuffled
    in unison* meaning that an element X from dataset and Y from labels must be shuffled to the same new positions within
    their respective datasets
    This helper method to dataset builder method

    Args:
        a: array,
        b: array,
    Returns:
        unison shuffled array
    """
    # check length match for input arrays
    assert len(a) == len(b)
    # create a random shuffling
    p = np.random.permutation(len(a))
    # return co-shuffled arrays a, b
    return a[p], b[p]


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

    # Init Var
    all_reports = []  # init an empty array that will store the privacy attack results
    batch_size = 50  # usually one of 32, 64, 128, ...
    epochs = 16
    epochs_per_report = 5  # how often should the privacy attacks be performed
    learning_rate = 0.001


    # Init dataset metadata
    dataset = 'cifar10'
    num_classes = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # CIFAR as numpy dict datatype, with three keys (id, iamge, label)
    # TF datasets come presplit, in this case CIFAR has 10k, 50k validation, test split
    train_ds = tf_datasets.as_numpy(
        tf_datasets.load(
            dataset,
            split=tf_datasets.Split.TRAIN,
            batch_size=-1,
        )
    )
    test_ds = tf_datasets.as_numpy(
        tf_datasets.load(
            dataset,
            split=tf_datasets.Split.TEST,
            batch_size=-1
        )
    )
    # extract images and labels from imported data, rescale them to support model convergence
    x_train = train_ds['image'].astype('float32') / 255.
    x_test = test_ds['image'].astype('float32') / 255.
    y_test_indices = test_ds['label'][:, np.newaxis]
    y_train_indices = train_ds['label'][:, np.newaxis]

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

    # # Train & Test: Densely Connected NN
    # callback = PrivacyMetrics(
    #     epochs_per_report,  # parameter that signals how often the privacy attacks should be run on the model
    #     "model_dense_layers"  # model codename internally
    # )
    # # write results of model fit (i.e. training) to var(history)
    # history_model_dense_layers = model_dense_layers.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs,
    #     callbacks=[callback],
    # )
    # # append results to all_reports array
    # all_reports.extend(callback.attack_results)
    #

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
