#!/usr/bin/python

from __future__ import print_function

import os
import math
import sys

from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

from create import create_linear_regressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if len(sys.argv) < 5 or len(sys.argv) > 6:
    print(sys.argv, len(sys.argv))
    print("Invalid arguments")
    sys.exit(1)

learning_rate = float(sys.argv[1])
steps = int(sys.argv[2])
batch_size = int(sys.argv[3])
regulation_strength = float(sys.argv[4])
data_file = sys.argv[5] if len(sys.argv) == 6 else "resources/ata.csv"

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

beatmaps_dataframe = pd.read_csv(data_file, sep=",")
beatmaps_dataframe = beatmaps_dataframe.reindex(np.random.permutation(beatmaps_dataframe.index))


def preprocess_features(beatmaps_dataframe):
    selected_features = beatmaps_dataframe[[
        "is_easy",
        "is_normal",
        "is_hard",
        "is_expert",
        "is_expert_plus",
        "length",
        "bpm",
        "note_jump_speed",
        "note_count",
        "bomb_count",
        "notes_per_second",
        "obstacle_count"
    ]]

    processed_features = selected_features.copy()
    return processed_features


def preprocess_targets(beatmaps_dataframe):
    output_targets = pd.DataFrame()
    output_targets["rating"] = beatmaps_dataframe["rating"]
    return output_targets


training_count = int(len(beatmaps_dataframe.index) * 0.75)
validation_count = len(beatmaps_dataframe.index) - training_count

training_examples = preprocess_features(beatmaps_dataframe.head(training_count))
training_targets = preprocess_targets(beatmaps_dataframe.head(training_count))

validation_examples = preprocess_features(beatmaps_dataframe.tail(validation_count))
validation_targets = preprocess_targets(beatmaps_dataframe.tail(validation_count))

print("Training examples:")
display.display(training_examples.describe())
print("Validation examples:")
display.display(validation_examples.describe())

print("Training targets:")
display.display(training_targets.describe())
print("Validation targets:")
display.display(validation_targets.describe())


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(feature) for feature in input_features])


def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}

    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(
        learning_rate,
        steps,
        batch_size,
        feature_columns,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets
):
    periods = 10
    steps_per_period = steps / periods

    linear_regressor = create_linear_regressor(learning_rate, feature_columns, "models/training", regulation_strength)

    training_input_fn = lambda: input_fn(
        training_examples,
        training_targets["rating"],
        batch_size=batch_size
    )
    predict_training_input_fn = lambda: input_fn(
        training_examples,
        training_targets["rating"],
        num_epochs=1,
        shuffle=False
    )
    predict_validation_input_fn = lambda: input_fn(
        validation_examples,
        validation_targets["rating"],
        num_epochs=1,
        shuffle=False
    )

    print("Training model...")
    print("RMSE:")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets)
        )

        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))

        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Training done.")

    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return linear_regressor


_ = train_model(
    learning_rate=learning_rate,
    steps=steps,
    batch_size=batch_size,
    feature_columns=construct_feature_columns(training_examples),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)
