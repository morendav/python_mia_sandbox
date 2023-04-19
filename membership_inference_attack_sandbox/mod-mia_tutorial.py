# copying this from https://colab.research.google.com/github/tensorflow/privacy/blob/master/g3doc/tutorials/privacy_report.ipynb


import numpy as np
from typing import Tuple
from scipy import special
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

# Set verbosity.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.exceptions import ConvergenceWarning

import warnings

warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
import tensorflow_privacy
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report



### NEW IMPORTS
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

import tensorflow_privacy
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackResultsCollection
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyMetric
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import PrivacyReportMetadata
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import privacy_report
# NEW IMPORTS






# Init Parameters
dataset = 'cifar10'
num_classes = 10
activation = 'relu'
num_conv = 3
batch_size = 20
epochs_per_report = 3
total_epochs = 24
lr = 0.001
all_reports = []



### NEW CODE - Import init var
### NEW CODE - Import init var
### NEW CODE - Import init var
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
img_height = 32  # original size 224, scaled to x
img_width = 32
epochs = 25
epochs_per_report = 3  # how often should the privacy attacks be performed
learning_rate = 0.0001
data_split = 0.3115  # percent split that will go to validation dataset.

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_directory_training,
    validation_split=data_split,
    subset="training",
    seed=33,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_directory_training,
    validation_split=data_split,
    subset="validation",
    seed=33,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
# Mine metadata from the imported training and validation batchdatasets data types
# image_dataset_from_directory utility will import a datatype _BatchDataSet which is not subscriptable
# this means that pulling sample images and their associated labels is not a matter of calling elements from this datatype
# _BatchDataset has some methods that are able to summarize the dataset, it's classes
# mine the class names
class_names = train_ds.class_names
num_classes = len(class_names)

# mine the dataset labels
# this creates an array of labels (numerical) size (1,N) where N is the number of samples in the dataset
train_label = np.concatenate([y for x, y in train_ds], axis=0)
test_label = np.concatenate([y for x, y in val_ds], axis=0)

# convert the (1,N) array into N (1) sized arrays (or matrix of size N,1) using numpy newaxis method
# this ensures that a single array (with a single value, for the class label)
# is passed to the MIA attackdata datatype
y_train_indices = train_label[:, np.newaxis]
y_test_indices = test_label[:, np.newaxis]

# preprocess training data in local cache
# this step is used to prevent i/o bottle neck
# if image size is restricted to smaller size this may not be issue
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

input_shape = (img_height, img_width, 3)  # hardcoded value for input shape, used by create cnn helper method
### NEW CODE - Import init var
### NEW CODE - Import init var
### NEW CODE - Import init var











# Defining some helper functions
def small_cnn(input_shape: Tuple[int],
              num_classes: int,
              num_conv: int,
              activation: str = 'relu') -> tf.keras.models.Sequential:
    """Setup a small CNN for image classification.

    Args:
      input_shape: Integer tuple for the shape of the images.
      num_classes: Number of prediction classes.
      num_conv: Number of convolutional layers.
      activation: The activation function to use for conv and dense layers.

    Returns:
      The Keras model.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Conv layers
    for _ in range(num_conv):
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation))
        model.add(tf.keras.layers.MaxPooling2D())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=activation))
    model.add(tf.keras.layers.Dense(num_classes))

    model.compile(
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    return model









class PrivacyMetrics(tf.keras.callbacks.Callback):
    def __init__(self, epochs_per_report, model_name):
        self.epochs_per_report = epochs_per_report
        self.model_name = model_name
        self.attack_results = []

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1

        if epoch % self.epochs_per_report != 0:
            return

        print(f'\nRunning privacy report for epoch: {epoch}\n')

        #### OLD CODE HERE
        # logits_train = self.model.predict(x_train, batch_size=batch_size)
        # logits_test = self.model.predict(x_test, batch_size=batch_size)
        # prob_train = special.softmax(logits_train, axis=1)
        # prob_test = special.softmax(logits_test, axis=1)
        #### OLD CODE HERE

        ### NEW CODE HERE
        logits_train = self.model.predict(train_ds, batch_size=batch_size)  # TODO - check if batch size here is needed
        logits_test = self.model.predict(val_ds, batch_size=batch_size)
        prob_train = special.softmax(logits_train, axis=1)  # TODO - check what axis is here
        prob_test = special.softmax(logits_test, axis=1)
        ### NEW CODE HERE


        # Add metadata to generate a privacy report.
        privacy_report_metadata = PrivacyReportMetadata(
            # Show the validation accuracy on the plot
            # It's what you send to train_accuracy that gets plotted.
            # accuracy_train=logs['val_accuracy'],  ## TODO - this is the tutorial version
            accuracy_train=logs['accuracy'],
            accuracy_test=logs['val_accuracy'],
            epoch_num=epoch,
            model_variant_label=self.model_name)

        attack_results = mia.run_attacks(
            AttackInputData(
                labels_train=y_train_indices[:, 0],
                labels_test=y_test_indices[:, 0],
                probs_train=prob_train,
                probs_test=prob_test
            ),
            SlicingSpec(
                entire_dataset=True,
                by_class=True
            ),
            attack_types=(
                AttackType.THRESHOLD_ATTACK,
                AttackType.LOGISTIC_REGRESSION
            ),
            privacy_report_metadata=privacy_report_metadata
        )
        self.attack_results.append(attack_results)
# Function to shuffle the datasets
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

















### CONTINUE - old tutorial code
### CONTINUE - old tutorial code
### CONTINUE - old tutorial code
# Creating and training models
# 2 layer depth
model_2layers = small_cnn(
    input_shape, num_classes, num_conv=2, activation=activation)
model_3layers = small_cnn(
    input_shape, num_classes, num_conv=3, activation=activation)









# # NEW CODE - TESTING if model definition from mia is what is leading to variable MIA
# # NEW CODE - TESTING if model definition from mia is what is leading to variable MIA
# # NEW CODE - TESTING if model definition from mia is what is leading to variable MIA
# # replace the conv2d helper method to make model with a hardcoded transplant from MIA script
# model_repeated_conv2d = Sequential([
#     layers.Input(shape=input_shape),
#     # layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
#     # layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(36, activation='relu'),
#     layers.Dense(num_classes)  # final layer must be the number of classes
# ])
# # model_2layers = model_repeated_conv2d
# model_2layers.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy']
# )
# # NEW CODE - TESTING if model definition from mia is what is leading to variable MIA
# # NEW CODE - TESTING if model definition from mia is what is leading to variable MIA
# # NEW CODE - TESTING if model definition from mia is what is leading to variable MIA









callback = PrivacyMetrics(epochs_per_report, "2 Layers")
history = model_2layers.fit(
    train_ds,
    # batch_size=batch_size,
    epochs=total_epochs,
    validation_data=val_ds,
    callbacks=[callback],
    shuffle=True,
)
all_reports.extend(callback.attack_results)


# # 3 layer depth
# callback = PrivacyMetrics(epochs_per_report, "3 Layers")
# history = model_3layers.fit(
#       x_train,
#       y_train,
#       batch_size=batch_size,
#       epochs=total_epochs,
#       validation_data=(x_test, y_test),
#       callbacks=[callback],
#       shuffle=True)
#
# all_reports.extend(callback.attack_results)

# append all results to results array
results = AttackResultsCollection(all_reports)

# create reports for MIA assessments
privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
epoch_plot = privacy_report.plot_by_epochs(
    results, privacy_metrics=privacy_metrics)

privacy_metrics = (PrivacyMetric.AUC, PrivacyMetric.ATTACKER_ADVANTAGE)
utility_privacy_plot = privacy_report.plot_privacy_vs_accuracy(
    results, privacy_metrics=privacy_metrics)

for axis in utility_privacy_plot.axes:
    axis.set_xlabel('Validation accuracy')


# plot accuracy and loss per epoch
epochs_range = range(total_epochs)
# Pull accuracy and loss per epoch from history, plot them for 2DConv
conv2D_acc = history.history['accuracy']
conv2D_val_acc = history.history['val_accuracy']
conv2D_loss = history.history['loss']
conv2D_val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, conv2D_acc, label='Training Accuracy')
plt.plot(epochs_range, conv2D_val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, conv2D_loss, label='Training Loss')
plt.plot(epochs_range, conv2D_val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
# plt.savefig(current_directory / 'conv2d_model_training_validation_accuracy.png')
plt.show()
