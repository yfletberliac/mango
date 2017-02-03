"""
This module constructs model_fn function that will be fed into the Estimator.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def model_fn(features, targets, mode, params, scope=None):
    embedding_size = params['embedding_size']
    vocab_size = params['vocab_size']
    max_sentence_length = params['max_sentence_length']
    max_story_length = params['max_story_length']
    hidden_units = params['hidden_units']
    debug = params['debug']

    story = features['story']  # 10 * 33

    batch_size = tf.shape(story)[0]
    normal_initializer = tf.random_normal_initializer(stddev=0.1)

    with tf.variable_scope(scope, 'Entity_Network', initializer=normal_initializer):
        # Embeddings taking care of paddings
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size)],
                                     dtype=tf.float32,
                                     shape=[vocab_size, 1])
        embedding_params_masked = embedding_params * embedding_mask  # 38 * 10
        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)  # 10 * 33 * 10

        # Recurrence
        cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        initial_state = cell.zero_state(batch_size, tf.float32)

        story_length = get_story_length(story_embedding)
        sentence_length = get_sentence_length(story_embedding)

        embedded_input = tf.reshape(story_embedding, [batch_size, max_sentence_length*max_story_length, embedding_size])

        outputs_rnn1, last_state = tf.nn.dynamic_rnn(cell, embedded_input,
                                                     sequence_length=story_length*max_sentence_length,
                                                     initial_state=initial_state)

        # Output
        output = get_output(last_state,
                            vocab_size=vocab_size,
                            initializer=normal_initializer
                            )
        prediction = tf.argmax(output, 1)

        # Training
        loss = get_loss(output, targets, mode)
        train_op = training_optimizer(loss, params, mode)

        if debug:
            tf.contrib.layers.summarize_tensor(sentence_length, 'sentence_length')
            tf.contrib.layers.summarize_tensor(story_length, 'story_length')
            tf.contrib.layers.summarize_tensor(outputs_rnn1, 'outputs_rnn1')
            tf.contrib.layers.summarize_tensor(last_state, 'last_state')
            tf.contrib.layers.summarize_tensor(output, 'output')
            tf.contrib.layers.summarize_variables()

            tf.add_check_numerics_ops()

        return prediction, loss, train_op


def get_story_length(sequence, scope=None):
    """
    Find the actual length of a story that has been padded with zeros.
    """
    with tf.variable_scope(scope, 'StoryLength'):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=[-1]))
        tmp = tf.reduce_max(used, reduction_indices=[-1])
        length = tf.cast(tf.reduce_sum(tmp, reduction_indices=[-1]), tf.int32)
        return length


def get_sentence_length(sequence, scope=None):
    """
    Find the actual length of a sentence that has been padded with zeros.
    """
    with tf.variable_scope(scope, 'SentenceLength'):  # 10 * 33 * 1O
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=[-1]))
        length = tf.cast(tf.reduce_sum(used, reduction_indices=[-1]), tf.int32)
        return length


def get_output(last_state, vocab_size, activation=tf.nn.relu, initializer=None, scope=None):
    with tf.variable_scope(scope, 'Output', initializer=initializer):
        _, embedding_size = last_state.get_shape().as_list()

        R = tf.get_variable('R', [embedding_size, vocab_size])
        H = tf.get_variable('H', [embedding_size, embedding_size])

        y = tf.matmul(activation(tf.matmul(last_state, H)), R)
        return y


def get_loss(output, labels, mode):
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return None
    return tf.contrib.losses.sparse_softmax_cross_entropy(output, labels)


def training_optimizer(loss, params, mode):
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        return None

    clip_gradients = params['clip_gradients']
    learning_rate_init = params['learning_rate_init']
    learning_rate_decay_rate = params['learning_rate_decay_rate']
    learning_rate_decay_steps = params['learning_rate_decay_steps']

    global_step = tf.contrib.framework.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        learning_rate=learning_rate_init,
        decay_steps=learning_rate_decay_steps,
        decay_rate=learning_rate_decay_rate,
        global_step=global_step,
        staircase=True)

    tf.contrib.layers.summarize_tensor(learning_rate, tag='learning_rate')

    train_op = tf.contrib.layers.optimize_loss(loss,
                                               global_step=global_step,
                                               learning_rate=learning_rate,
                                               optimizer='Adam',
                                               clip_gradients=clip_gradients
                                               )

    return train_op
