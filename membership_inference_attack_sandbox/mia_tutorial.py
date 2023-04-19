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


# Init Parameters
dataset = 'cifar10'
num_classes = 10
activation = 'relu'
num_conv = 3
batch_size = 20
epochs_per_report = 5
total_epochs = 16
lr = 0.001
all_reports = []






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
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
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

        logits_train = self.model.predict(x_train, batch_size=batch_size)
        logits_test = self.model.predict(x_test, batch_size=batch_size)

        prob_train = special.softmax(logits_train, axis=1)
        prob_test = special.softmax(logits_test, axis=1)

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








'''
LOADING THE DATASET

There are explicit parameters for each of training and test datasets
X = the variables, to be labeled (input data)
Y = the labels, target of classification algorithm
'''

print('Loading the dataset.')
train_ds = tfds.as_numpy(
    tfds.load(dataset, split=tfds.Split.TRAIN, batch_size=-1))
test_ds = tfds.as_numpy(
    tfds.load(dataset, split=tfds.Split.TEST, batch_size=-1))
# extract images and labels from imported data
# note that the datasets are already rescaled
x_train = train_ds['image'].astype('float32') / 255.
y_train_indices = train_ds['label'][:, np.newaxis]
x_test = test_ds['image'].astype('float32') / 255.
y_test_indices = test_ds['label'][:, np.newaxis]

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train_indices, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_indices, num_classes)

input_shape = x_train.shape[1:]

assert x_train.shape[0] % batch_size == 0, "The tensorflow_privacy optimizer doesn't handle partial batches"

# class names for cifar dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']











# ### MODIFYING THE DATASETS
# ### MODIFYING THE DATASETS
# ### MODIFYING THE DATASETS
# # ## PLOT THE PRE-Shuffled and modded dataset
# # plt.figure(figsize=(10, 10))
# # for i in range(25):
# #     plt.subplot(5, 5, i + 1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.grid(False)
# #     plt.imshow(x_train[i])
# #     # The CIFAR labels happen to be arrays,
# #     # which is why you need the extra index
# #     plt.xlabel(class_names[y_train_indices[i][0]])
# # plt.show()
# # NEW CODE - SELECT ONLY SOME LABELS
# num_classes = 3
# # hypothesis: fewer classes makes the MIA test more variable
# train_ind_0 = np.where(y_train_indices == 0)[0]
# train_ind_1 = np.where(y_train_indices == 1)[0]
# train_ind_2 = np.where(y_train_indices == 2)[0]
# # pull out input data using indices for classes defined above.
# # we will discard the validation dataset all together for now
# x_0 = x_train[(train_ind_0), :, :, :]
# x_1 = x_train[(train_ind_1), :, :, :]
# x_2 = x_train[(train_ind_2), :, :, :]
# # pull out labels from the training dataset for the classes we will use only
# train_label_0 = y_train_indices[train_ind_0]
# train_label_1 = y_train_indices[train_ind_1]
# train_label_2 = y_train_indices[train_ind_2]
# # concatenate only the three classes of CIFAR we will use
# new_data_3classes = np.concatenate((x_0, x_1, x_2))
# new_data_labels_3classes = np.concatenate((train_label_0, train_label_1, train_label_2))
#
# shuffled_new_data_3classes, shuffled_new_data_labels_3classes = unison_shuffled_copies(new_data_3classes,
#                                                                                        new_data_labels_3classes)
# # Update variables used in rest of program to point to new 3class dataset
# # there are 5000 examples of each class, there are now three classes meaning we have 15,000 images.
# x_train = shuffled_new_data_3classes[0:10000, :, :, :]
# y_train_indices = shuffled_new_data_labels_3classes[0:10000]
# x_test = shuffled_new_data_3classes[10001:, :, :, :]
# y_test_indices = shuffled_new_data_labels_3classes[10001:]
#
#
# # NEW CODE - DATASET MOD - DOWNSAMPLE THE WHOLE DATASET
# # NEW CODE - DATASET MOD - DOWNSAMPLE THE WHOLE DATASET
# # NEW CODE - DATASET MOD - DOWNSAMPLE THE WHOLE DATASET
# # test to determine if smaller test and training datasets resulted in MIA attacks having more variability
# # hypothesis: smaller datasets means the models have lower losses per epoch,
# # model learns quicker to fit to data, with limited data. Lower losses result in instable MIA results
# x_train = x_train[1:301]
# y_train_indices = y_train_indices[1:301]
# x_test = x_test[1:201]
# y_test_indices = y_test_indices[1:201]
#
# # # NEW CODE - SHOW THE DATASETS AGAIN
# # # NEW CODE - SHOW THE DATASETS AGAIN
# # # NEW CODE - SHOW THE DATASETS AGAIN
# # # if codeblocks above are uncomments then the dataset is modified to be 3 classes only, and is much smaller,
# # # multiple orders of magnitude smaller
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_train[i])
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[y_train_indices[i][0]])
# plt.show()
# # CODE BLOCK - repeated from the tutorial without modifications
# # this section is required preprocessing for model training
# # Convert class vectors to binary class matrices.
# # this is only used during the model fit method by keras
# y_train = tf.keras.utils.to_categorical(y_train_indices, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test_indices, num_classes)
#
# # used in the CNN creator method
# input_shape = x_train.shape[1:]
# # mia doesn't support partial batch handling, so this checks if there are any partial batches. otherwise unused
# assert x_train.shape[0] % batch_size == 0, "The tensorflow_privacy optimizer doesn't handle partial batches"
#
# ### MODIFYING THE DATASETS
# ### MODIFYING THE DATASETS
# ### MODIFYING THE DATASETS















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
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=total_epochs,
    validation_data=(x_test, y_test),
    callbacks=[callback],
    shuffle=True)
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
