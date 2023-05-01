"""
Introduction to Membership Inference Attack - Privacy testing for ML Models
Version 2.1
    * see mia sandbox > ... > mia_intro_image_class.py for original WIP version
    * improvements over 1.1 (WIP) include model builder methods, and a method to slice training data into M of N classes

This intro will perform an MIA privacy analysis on models of various architectures and sizes, as well as the number of
classes in the source training/validation datasets
Training accuracy and loss vs validation accuracy and loss will be shown plotted over epochs in model training
Over the total number of epochs all models are expected to overfit to the training dataset

The expected results should show:
1. Models overfit as the number of epochs increases
2. Different models, with different architectures and different capacities, overfit at different rates
3. Overfitting and model memorization measurements are covaried

The experiments are run on three basis of comparison
1. Using a 10class dataset, two models of equal depth but different architecture (Convolutional vs Dense) are tested
2. Using the same 10class dataset, two Dense neural net models of different depths are tested (6+1 vs 3+1) layers
   * NOTE: model builder methods add an additional layer between output layer and for _ in range loop, thus N+1 layers
3. Using a downsampling of the 10class to use only the first 4 classes. Models from experiment 2 are recreated
   * NOTE: the method for dataset truncation is lazy, does not do a random sampling instead just picks the first N of M

Dataset credit: kudos to TF public dataset CIFAR10

Copyright (c) 2023, d.l.moreno
All rights reserved

This source code is licensed under the Apache v2 license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple, Union, Any

# Standard & third party libraries
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import special
from pathlib import Path

# Import tensorflow, tf datasetss, and tensorflow privacy libraries and utilities
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_datasets as tf_datasets
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report

# Set verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.exceptions import ConvergenceWarning
# suppress convergence warnings for the logistic regression MIA
import warnings

warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


# Define MIA helper methods
# builder methods are used to create models with some arbitrary values for
# their characteristics (e.g. depth, number of convolutions, etc)
def conv_nn_builder(input_shape: Tuple[int],
                    num_classes: int,
                    depth: int,
                    neurons_per_layer: int = 64,
                    activation: str = 'relu'
                    ) -> tf.keras.models.Sequential:
    """
    Helper Method to build a multi-layered conv2d model of arbitrary depth (depth).
    The filter size for each conv2D is not parameterized. No padding per layer

    Args:
        input_shape: Integer tuple, shape of a sample image. e.g. (64,64,3)
        num_classes: Output, number of classes
        neurons_per_layer: number of neurons in each layer. Default 64 neurons
        depth: Depth of conv2D neural net
        activation: The activation function to use for conv and dense layers. Default RELU
    Returns:
        TF.Keras compiled model with the parameters passed to this nn builder method
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
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


def dense_nn_builder(input_shape: Tuple[int],
                     num_classes: int,
                     depth: int,
                     neurons_per_layer: int = 64,
                     activation: str = 'relu'
                     ) -> tf.keras.models.Sequential:
    """
    Helper Method to build a multi-layered densely connected NN model of arbitrary depth (depth)


    Args:
        input_shape: Integer tuple, shape of a sample image. e.g. (64,64,3)
        num_classes: Output, number of classes
        neurons_per_layer: number of neurons in each layer. Default 64 neurons
        depth: Depth of densely connected neural networks
        activation: The activation function to use for conv and dense layers. Default RELU
    Returns:
        TF.Keras compiled model with the parameters passed to this nn builder method
    """
    model = Sequential()
    model.add(layers.Flatten(input_shape=input_shape))

    # loop to create hidden layers of model of arbitrary depth
    # each layer has neurons_per_layer neurons, which is default 64 unless specified at method call
    for _ in range(depth):
        model.add(layers.Dense(neurons_per_layer, activation=activation))
    model.add(layers.Dense(neurons_per_layer, activation=activation))  # conv2d has an additional hidden layer here
    # adding this (above) additional hidden layer to make this a fair comparison
    model.add(layers.Dense(num_classes))  # logits must equal number of class in prediction space
    model.compile(
        # Depending on the format of the labels numpy array (N:1, or 1:N) you will have to swap these definitions
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


# TODO dont be lazy about returned datatypes here, come back to this later
def dataset_builder(dataset: dict,
                    num_classes: int = 10,
                    ):
    # ) -> tuple[Union[float, Any], Any, Any]:
    """
    Build a model-ready dataset for training and validation from an input dataset dictionary
    Assumes the dictionary passed is generated from TensorFlow datasets library,
    and has three keys: [image, label, id].

    Args:
        dataset: dict,
        num_classes: int = 10,
    Returns:
        ds_images images in dataset
        ds_labels labels of images in dataset
        ds_labels_training_vector sparse representation of the labels, length number of classes, with 1 nonzero entry
    """

    # preprocess dataset and extract dataset input, labels, and create a sparse vector
    # representation of the labels for training
    # First, preprocess the images into float typed entries, rescale to support model convergence
    ds_images = dataset['image'].astype('float32') / 255.
    # extract labels from dataset
    ds_labels = dataset['label'][:, np.newaxis]

    # logic to gate dataset truncation conditionally on whether the passed parameter is not equal to the number
    # of classes the dataset already has
    if num_classes != np.unique(dataset['label']).size:
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
            labels_for_i_class = ds_labels[indices_for_i_class]  # only label values for the current index

            # append to a growing dataset and labels array
            if i != 0:
                new_truncated_dataset = np.concatenate((new_truncated_dataset, images_for_i_class))
                new_truncated_labels = np.concatenate((new_truncated_labels, labels_for_i_class))
            else:
                new_truncated_dataset = images_for_i_class
                new_truncated_labels = labels_for_i_class

        # after the new dataset and labels truncated arrays are created they must be shuffled
        # if they are not shuffled then the model will skew towards the first class since all examples
        # it learns in the first passes are of class 0
        ds_images, ds_labels = unison_shuffle_arrays(
            new_truncated_dataset,
            new_truncated_labels,
        )
    # create vector per image representing a sparse vector with a 1 value at the index for the label
    ds_labels_training_vector = tf.keras.utils.to_categorical(ds_labels, num_classes)

    return ds_images, ds_labels, ds_labels_training_vector


def unison_shuffle_arrays(a, b):
    """
    Shuffle two arrays of equal length
    Used to shuffle dataset and label vectors during truncation, note that both dataset and labels *must be shuffled
    in unison* meaning that an element X from dataset and element Y from labels must be shuffled to the same
    new positions within their respective datasets. This helper method to dataset builder method.
    This will fail if length(a) is not equal to length(b)

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

    E is defined as a parameter within the program (see init var section in main)
    This class is interpreted at each epoch of training, however only at every E epochs are the tests done.
    This is handled using modulo (current_epoch modulo epoch_test)

    Required data objects from main program include: model, named model (passed in callback method),
    training and validation datasets, as well as labels for training and validation datasets

    Credit here is due in large part to TensorFlow Privacy github for great training and documentation :D
    """

    def __init__(self, epochs_per_report, model_name, train_input, train_labels_indices, val_input, val_labels_indices):
        # when initiatlizing class, assign characteristics to the class
        self.epochs_per_report = epochs_per_report
        self.model_name = model_name
        # these are used for MIA, will be new for each training run
        self.training_input = train_input
        self.training_labels_ind = train_labels_indices
        self.validation_input = val_input
        self.validation_labels_ind = val_labels_indices
        self.attack_results = []

    def on_epoch_end(self, epoch, logs=None):
        # increment the epoch variable by one at each training epoch
        epoch = epoch + 1

        # if current epoch modulo epochs per report is not exactly zero exit
        if epoch % self.epochs_per_report != 0:
            return

        print(f'\n\nEvaluating model by running a Membership Inference Attack')
        print(f'Callback at training epoch {epoch}\n')

        # capture logits and convert to probabilities by applying softmax function to logits.
        # Uses class variables for training and validation datasets
        logits_train = self.model.predict(
            self.training_input,
            batch_size=batch_size,
        )
        logits_test = self.model.predict(
            self.validation_input,
            batch_size=batch_size,
        )
        prob_train = special.softmax(logits_train, axis=1)
        prob_test = special.softmax(logits_test, axis=1)

        # Add metadata to generate a privacy report.
        privacy_report_metadata = PrivacyReportMetadata(
            # Show the validation accuracy on the plot
            # It's what you send to train_accuracy that gets plotted.
            accuracy_train=logs['val_accuracy'],
            # accuracy_train=logs['accuracy'],
            accuracy_test=logs['val_accuracy'],
            epoch_num=epoch,  # current epoch at privacy testing
            model_variant_label=self.model_name,
        )

        # run the attacks
        attack_results = mia.run_attacks(
            AttackInputData(
                # labels must be passed, and must of dimension (N,1)
                # preprocessing to configure the labels matrices is done as a preprocessing step within the main program
                labels_train=self.training_labels_ind[:, 0],
                labels_test=self.validation_labels_ind[:, 0],
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
    reports_10class_modelType = []  # data structure to compare model types (Conv2d vs Dense) for the 10class dataset
    reports_10class_denseModelDepth = []  # data structure to compare model depth for Dense NN model for 10class dataset
    reports_4class_denseModelDepth = []  # data structure to compare model depth for Dense NN model for 4class dataset
    batch_size = 50  # usually one of 32, 64, 128, ... dataset sample count must be entirely divisible by batch size
    epochs = 30  # total number of epochs the experiment will run for
    epochs_range = range(epochs)  # used for plotting figures later, assumes all models are trained for E epochs
    epochs_per_report = 2  # how often should the privacy attacks be performed
    learning_rate = 0.001  # hyperparameter for all models
    # Init directory variables
    current_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    # Init dataset metadata
    dataset = 'cifar10'
    num_classes = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # CIFAR as numpy dict datatype, with three keys (id, image, label)
    # TF datasets come presplit, in this case CIFAR has 10k, 50k validation, test split
    train_ds = tf_datasets.as_numpy(
        tf_datasets.load(
            dataset,
            split=tf_datasets.Split.TRAIN,
            batch_size=-1,
        )
    )
    validation_ds = tf_datasets.as_numpy(
        tf_datasets.load(
            dataset,
            split=tf_datasets.Split.TEST,
            batch_size=-1
        )
    )

    # Create training and test datasets with 10 (default number) of classes
    # use the dataset_builder helper method to create the training and validation datasets
    # x_indices variables are used for MIA callbacks
    # x_x and x_y arrays are images and sparse vector labels respectively
    train_10_x, train_10_y_indices, train_10_y = dataset_builder(
        dataset=train_ds,
    )
    validation_10_x, validation_10_y_indices, validation_10_y = dataset_builder(
        dataset=validation_ds,
    )
    # Create training and test datasets with 4 classes
    train_4_x, train_4_y_indices, train_4_y = dataset_builder(
        dataset=train_ds,
        num_classes=4,
    )
    validation_4_x, validation_4_y_indices, validation_4_y = dataset_builder(
        dataset=validation_ds,
        num_classes=4,
    )

    # validate that the 10class and 4class datasets will not leave any partial batches
    # MIA tensorflow privacy library does not handle partial batches
    assert train_10_x.shape[0] % batch_size == 0, "10Class partial batch, error in tensorflow_privacy optimizer"
    assert train_4_x.shape[0] % batch_size == 0, "4Class partial batch, error in tensorflow_privacy optimizer"

    # Generate the compiled models
    # Experiment: Model Type trained on 10class dataset
    # generate two different types of models to compare MIA across Convolutional NN and Dense NN
    conv_3layer_10label = conv_nn_builder(
        input_shape=train_10_x.shape[1:],
        num_classes=10,
        neurons_per_layer=64,
        depth=3,
        activation='relu',
    )
    dense_3layer_10label = dense_nn_builder(
        input_shape=train_10_x.shape[1:],
        num_classes=10,
        neurons_per_layer=64,
        depth=3,
        activation='relu',
    )

    # Experiment: Model Depth trained on 10class dataset. Both models are Dense NN
    # Note, the 3+1 depth model was created for experiment 1, this will not be recreated again
    dense_6layer_10label = dense_nn_builder(
        input_shape=train_10_x.shape[1:],
        num_classes=10,
        neurons_per_layer=64,
        depth=6,
        activation='relu',
    )

    # Experiment: Compare Experiment 1 models trained on a 4 class dataset (4 first classes from 10class dataset)
    # these models cannot be reused from experiment 1 because we need 4 logtis here instead of 10 from experiment 1
    dense_6layer_4label = dense_nn_builder(
        input_shape=train_4_x.shape[1:],
        num_classes=4,
        neurons_per_layer=64,
        depth=6,
        activation='relu',
    )
    dense_3layer_4label = dense_nn_builder(
        input_shape=train_4_x.shape[1:],
        num_classes=4,
        neurons_per_layer=64,
        depth=3,
        activation='relu',
    )

    print(f'\n\nReport model architectures')
    conv_3layer_10label.summary()
    dense_3layer_10label.summary()
    dense_6layer_10label.summary()
    dense_6layer_4label.summary()
    dense_3layer_4label.summary()

    # Train Models and run MIA while doing so via a callback
    print(f'\n\nTrain models and test at frequency: every {epochs_per_report} epoch ]\n')
    # build comparison between model types for 10class dataset, start with conv2d model and proceed to dense nn model
    # train and append MIA report for 10class dense NN 3  layer
    callback = PrivacyMetrics(
        epochs_per_report=epochs_per_report,
        model_name="denseNN_3layer",
        train_input=train_10_x,
        train_labels_indices=train_10_y_indices,
        val_input=validation_10_x,
        val_labels_indices=validation_10_y_indices,
    )
    # naming convention: history_(number of classes)_(which experiment)_(which variable)
    history_10class_modelType_dense = dense_3layer_10label.fit(
        train_10_x,
        train_10_y,
        batch_size=batch_size,
        validation_data=(validation_10_x, validation_10_y),
        epochs=epochs,
        callbacks=[callback],
    )
    reports_10class_modelType.extend(callback.attack_results)  # add results to the model type 10class dataset results
    reports_10class_denseModelDepth.extend(callback.attack_results)  # add results to the model depth 10c dataset result
    # train and append MIA report for 10class conv2d 3  layer
    callback = PrivacyMetrics(
        epochs_per_report=epochs_per_report,  # signals how often the privacy attacks should be run on the model
        model_name="conv2d_3layer",  # model codename internally
        train_input=train_10_x,
        train_labels_indices=train_10_y_indices,
        val_input=validation_10_x,
        val_labels_indices=validation_10_y_indices,
    )
    history_10class_modelType_conv2d = conv_3layer_10label.fit(
        train_10_x,
        train_10_y,
        batch_size=batch_size,
        validation_data=(validation_10_x, validation_10_y),
        epochs=epochs,
        callbacks=[callback],
    )
    reports_10class_modelType.extend(callback.attack_results)  # build comparison for 10class model types

    # build comparison between model depths for the 10class dataset, for dense NN of depth 3 and 6
    # train and append MIA report for 10class dense 6layer depth model
    callback = PrivacyMetrics(
        epochs_per_report=epochs_per_report,
        model_name="denseNN_6layer",
        train_input=train_10_x,
        train_labels_indices=train_10_y_indices,
        val_input=validation_10_x,
        val_labels_indices=validation_10_y_indices,
    )
    history_10class_modelType_dense6L = dense_6layer_10label.fit(
        train_10_x,
        train_10_y,
        batch_size=batch_size,
        validation_data=(validation_10_x, validation_10_y),
        epochs=epochs,
        callbacks=[callback],
    )
    reports_10class_denseModelDepth.extend(callback.attack_results)

    # build comparison between model depth for 4class datasets
    # train and append MIA report for 4class 3layer dense NN
    callback = PrivacyMetrics(
        epochs_per_report=epochs_per_report,
        model_name="4c_denseNN_3layer",
        train_input=train_4_x,
        train_labels_indices=train_4_y_indices,
        val_input=validation_4_x,
        val_labels_indices=validation_4_y_indices,
    )
    history_4class_depth_3layer = dense_3layer_4label.fit(
        train_4_x,
        train_4_y,
        batch_size=batch_size,
        validation_data=(validation_4_x, validation_4_y),
        epochs=epochs,
        callbacks=[callback],
    )
    reports_4class_denseModelDepth.extend(callback.attack_results)
    # train and append MIA report for 10class conv2d 3  layer
    callback = PrivacyMetrics(
        epochs_per_report=epochs_per_report,
        model_name="4c_denseNN_6layer",
        train_input=train_4_x,
        train_labels_indices=train_4_y_indices,
        val_input=validation_4_x,
        val_labels_indices=validation_4_y_indices,
    )
    history_4class_depth_6layer = dense_6layer_4label.fit(
        train_4_x,
        train_4_y,
        batch_size=batch_size,
        validation_data=(validation_4_x, validation_4_y),
        epochs=epochs,
        callbacks=[callback],
    )
    reports_4class_denseModelDepth.extend(callback.attack_results)

    # Generate plots for experiment result visualization & demo sampling of the datasets
    # Report MIA results for model experiments
    privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)  # define report types for all experiments

    # Sxs Reporting - Model Type experiments
    # Model loss/accuracy vs epoch, model MIA results vs epoch. 10class dataset, dense vs conv2d
    # Plot MIA results
    expModel_results = AttackResultsCollection(reports_10class_modelType)
    # plotting makes use of built in privacy testing plot method (from imported library)
    # documentation found in:
    # github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/privacy_report.py
    epoch_plot = privacy_report.plot_by_epochs(
        expModel_results,
        privacy_metrics=privacy_metrics
    )
    epoch_plot.savefig(current_directory / 'expModelType_mia_results.png')
    # Plot model training accuracy & loss vs training epoch
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, history_10class_modelType_conv2d.history['accuracy'], color='blue', label='Conv Train')
    plt.plot(epochs_range, history_10class_modelType_conv2d.history['val_accuracy'], color='black', label='Conv Val')
    plt.plot(epochs_range, history_10class_modelType_dense.history['accuracy'], color='blue', linestyle='dashed', label='Dense Train')
    plt.plot(epochs_range, history_10class_modelType_dense.history['val_accuracy'], color='black', linestyle='dashed', label='Dense Val')
    plt.legend(loc='lower right')
    plt.title('ModelType Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, history_10class_modelType_conv2d.history['loss'], color='blue', label='Conv Train')
    plt.plot(epochs_range, history_10class_modelType_conv2d.history['val_loss'], color='black', label='Conv Val')
    plt.plot(epochs_range, history_10class_modelType_dense.history['loss'], color='blue', linestyle='dashed', label='Dense Train')
    plt.plot(epochs_range, history_10class_modelType_dense.history['val_loss'], color='black', linestyle='dashed', label='Dense Val')
    plt.legend(loc='upper right')
    plt.title('ModelType Loss')
    plt.savefig(current_directory / 'expModelType_training_results.png')
    plt.show()

    # Sxs Reporting - Model Depth, both dense models
    # Model loss/accuracy vs epoch, model MIA results vs epoch. 10class dataset, dense3layer vs dense6layer
    # note each dense model has an additional layer, so 6+1 = 7 vs 3+1 =4 (see model builder method for dense)
    # Plot MIA results
    expDepth6_results = AttackResultsCollection(reports_10class_denseModelDepth)
    epoch_plot = privacy_report.plot_by_epochs(
        expDepth6_results,
        privacy_metrics=privacy_metrics
    )
    epoch_plot.savefig(current_directory / 'expDepth_mia_results.png')
    # Plot model training accuracy & loss vs training epoch
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, history_10class_modelType_dense6L.history['accuracy'], color='blue', label='7Layer Train')
    plt.plot(epochs_range, history_10class_modelType_dense6L.history['val_accuracy'], color='black', label='7Layer Val')
    plt.plot(epochs_range, history_10class_modelType_dense.history['accuracy'], color='blue', linestyle='dashed', label='4Layer Train')
    plt.plot(epochs_range, history_10class_modelType_dense.history['val_accuracy'], color='black', linestyle='dashed', label='4Layer Val')
    plt.legend(loc='lower right')
    plt.title('ModelDepth Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, history_10class_modelType_dense6L.history['loss'], color='blue', label='7Layer Train')
    plt.plot(epochs_range, history_10class_modelType_dense6L.history['val_loss'], color='black', label='7Layer Val')
    plt.plot(epochs_range, history_10class_modelType_dense.history['loss'], color='blue', linestyle='dashed', label='4Layer Train')
    plt.plot(epochs_range, history_10class_modelType_dense.history['val_loss'], color='black', linestyle='dashed', label='4Layer Val')
    plt.legend(loc='upper right')
    plt.title('ModelDepth Loss')
    plt.savefig(current_directory / 'expDepth_training_results.png')
    plt.show()

    # Sxs Reporting - 4class dataset, model type experiments
    # Model loss/accuracy vs epoch, model MIA results vs epoch. 4class dataset, dense3layer vs conv2d 3 layer
    # Plot MIA results
    exp4class_results = AttackResultsCollection(reports_4class_denseModelDepth)
    epoch_plot = privacy_report.plot_by_epochs(
        exp4class_results,
        privacy_metrics=privacy_metrics
    )
    epoch_plot.savefig(current_directory / 'exp4Class_mia_results.png')
    # Plot model training accuracy & loss vs training epoch
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history_4class_depth_3layer.history['accuracy'], color='blue', label='4Layer Train')
    plt.plot(epochs_range, history_4class_depth_3layer.history['val_accuracy'], color='black', label='4Layer Val')
    plt.plot(epochs_range, history_4class_depth_6layer.history['accuracy'], color='blue', linestyle='dashed', label='7Layer Train ')
    plt.plot(epochs_range, history_4class_depth_6layer.history['val_accuracy'], color='black', linestyle='dashed', label='7Layer Val')
    plt.legend(loc='lower right')
    plt.title('4Class Accuracy')
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history_4class_depth_3layer.history['loss'], color='blue', label='4Layer Train')
    plt.plot(epochs_range, history_4class_depth_3layer.history['val_loss'], color='black', label='4Layer Val')
    plt.plot(epochs_range, history_4class_depth_6layer.history['loss'], color='blue', linestyle='dashed', label='7Layer Train')
    plt.plot(epochs_range, history_4class_depth_6layer.history['val_loss'], color='black', linestyle='dashed', label='7Layer Val')
    plt.legend(loc='upper right')
    plt.title('4Class Loss')
    plt.savefig(current_directory / 'exp4Class_training_results.png')
    plt.show()

    # plot sampling of the 10-class dataset
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_10_x[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_10_y_indices[i][0]])
    plt.show()
    plt.savefig(current_directory / 'dataset_10class_cifar.png')
    # plot sampling of the 4-class dataset
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_4_x[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_4_y_indices[i][0]])
    plt.show()
    plt.savefig(current_directory / 'dataset_4class_cifar.png')