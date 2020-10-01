# Copyright 2020 The PEGASUS Authors..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Library for creating tf.data infeeds."""

from pegasus.data import all_datasets
from pegasus.ops import public_parsing_ops
import tensorflow as tf


def get_input_fn(parser_fn,
                 input_pattern,
                 mode,
                 prefetch=True,
                 drop_remainder=True,
                 parallelism=32):
  """Estimator input_fn for TFRecords."""

  # Parser can support any input types -- e.g. tfexamples, or arbitrary
  # protobufs. The parser_fn also returns expected shapes, as a map from
  # feature_name to feature_shape (list of ints).
  parser, shapes = parser_fn(mode=mode)
  training = mode == tf.estimator.ModeKeys.TRAIN
  if not training:
    parallelism = 1

  def input_fn(params):
    """Input function."""
    dataset = all_datasets.get_dataset(input_pattern, training)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& This is the dataset &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(dataset)
    print(type(dataset))
    
    tf.config.experimental_run_functions_eagerly(True)
    def show_batch(dataset):
      for batch, label in dataset.take(1):
        for key, value in batch.items():
          print("{:20s}: {}".format(key,value.numpy()))

    print(show_batch(dataset))
    
    tf.config.experimental_run_functions_eagerly(False)
    
    dataset = dataset.map(parser, num_parallel_calls=parallelism)
    dataset = dataset.unbatch() #Splits elements of a dataset into multiple elements.
    if training:
#       dataset = dataset.shuffle(10000) #Randomly shuffles the elements of this dataset.
      dataset = dataset.repeat(100) #Repeats this dataset count times. The default behavior (if count is None or -1) is for the dataset be repeated indefinitely.
    dataset = dataset.padded_batch(
        params["batch_size"],
        padded_shapes=shapes,
        drop_remainder=drop_remainder)
    if prefetch:
      dataset = dataset.prefetch(512)
    return dataset

  return input_fn


def serving_input_fn(params):
  """Returns expected input spec for exported savedmodels."""
  inputs_ph = tf.placeholder(
      dtype=tf.string, shape=[params.batch_size], name="inputs")

  inputs = public_parsing_ops.encode(inputs_ph, params.max_input_len,
                                     params.vocab_filename, params.encoder_type,
                                     params.length_bucket_size > 0)
  inputs = tf.reshape(inputs, [params.batch_size, params.max_input_len])
  features = {"inputs": inputs}
  return tf.estimator.export.ServingInputReceiver(
      features=features, receiver_tensors=inputs_ph)
