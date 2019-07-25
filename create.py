import tensorflow as tf


def create_linear_regressor(learning_rate, feature_columns, model_dir):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    return tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=optimizer,
        model_dir=model_dir
    )
