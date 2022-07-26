from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

# - linear Regression -
# data
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # eval data
# print(dftrain.head())
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# print(dftrain.shape)
# print(dftrain.describe())
CATEGORICAL_COLUMNS = ['sex', 'class', 'deck',  # Categorical is anything non-numeric
                       'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare', 'parch', 'n_siblings_spouses']

feature_columns = []
# before train model, convert categorical into numeric data
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from column
    # adding to feature columns after converting to numeric value with vocabulary list
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


# - training process -
# model data is going to be streamed into it in small batches of 32.
# An epoch is simply one stream of our entire dataset.
# An epoch is the amount of times our model will see the entire dataset
# Since we need to feed our data in batches and multiple times,
# we need to create something called an input function.
# The input function simply defines how our dataset will
# be converted into batches at each epoch.

# The TensorFlow model we are going to use requires that the data\
# we pass it comes in as a tf.data.Dataset object. This means we must
# create a input function that can convert our current
# pandas dataframe into that object.


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices(
            (dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(
            num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset

    return input_function  # return a function object for use


train_input_fn = make_input_fn(dftrain, y_train)
# here we will call the input_function that was
# returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# use a linear estimator to utilize the linear regression algorithm.
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
# We create a linear estimator by passing the feature columns we created earlier
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

clear_output()  # clears console output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model

# we can actually use this model to make predictions. with the
# predict() method
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
