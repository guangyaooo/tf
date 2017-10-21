# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def test2():
    # Import data
    data_sets = input_data.read_data_sets("MNIST_data", one_hot=True)

    # Create the model
    images_placeholder = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(images_placeholder, W) + b

    # Define loss and optimizer
    labels_placeholder = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = data_sets.train.next_batch(100)
        sess.run(train_step, feed_dict={images_placeholder: batch_xs, labels_placeholder: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={images_placeholder: data_sets.test.images,
                                        labels_placeholder: data_sets.test.labels}))

    test_images, test_labels = data_sets.test.next_batch(data_sets.test.num_examples)
    test_data = sess.run(logits, feed_dict={images_placeholder: test_images, labels_placeholder: test_labels})
    # tf.nn.softmax(test_data)
    test_data = sess.run(tf.nn.softmax(test_data))

    train_images, trian_labels = data_sets.train.next_batch(data_sets.train.num_examples)
    train_data = sess.run(logits, feed_dict={images_placeholder: train_images, labels_placeholder: trian_labels})
    # tf.nn.softmax(train_data)
    train_data = sess.run(tf.nn.softmax(train_data))

    return train_data, test_data, trian_labels, test_labels
