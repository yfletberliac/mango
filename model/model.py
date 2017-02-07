"""
This module constructs model_fn that will be fed into the Estimator.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def model_fn(features, targets, mode, params, scope=None):
    embedding_size = params['embedding_size']
    batch_size_int = params['batch_size_int']
    vocab_size = params['vocab_size']
    max_story_char_length = params['max_story_char_length']
    max_story_word_length = params['max_story_word_length']
    token_space = params['token_space']
    token_sentence = params['token_sentence']
    hidden_units = params['hidden_units']
    debug = params['debug']

    story = features['story']  # ? * 1 * 311

    batch_size = tf.shape(story)[0]
    normal_initializer = tf.random_normal_initializer(stddev=0.1)

    with tf.variable_scope(scope, 'Mango', initializer=normal_initializer):
        # Embeddings taking care of paddings
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size)],
                                     dtype=tf.float32,
                                     shape=[vocab_size, 1])
        embedding_params_masked = embedding_params * embedding_mask  # 38 * embedding_size
        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)  # ? * 1 * 311 * embedding_size

        indices_word = get_word_indices(story, token_space)  # index(words) # TODO add the dots for the missing words
        indices_sentence = get_sentence_indices(story, token_sentence)  # index(sentences)

        # Recurrence
        story_length = get_story_char_length(story_embedding)
        word_length = get_story_word_length(story, token_space, vocab_size, embedding_size)
        sentence_length = get_story_sentence_length(story, token_sentence, vocab_size, embedding_size)

        embedded_input = tf.squeeze(story_embedding, [1])
        # TODO remove non-linearities between layers
        cell_1 = tf.nn.rnn_cell.GRUCell(hidden_units)
        cell_2 = tf.nn.rnn_cell.GRUCell(hidden_units)
        cell_3 = tf.nn.rnn_cell.GRUCell(hidden_units)
        initial_state_1 = cell_1.zero_state(batch_size, tf.float32)
        initial_state_2 = cell_2.zero_state(batch_size, tf.float32)
        initial_state_3 = cell_3.zero_state(batch_size, tf.float32)

        with tf.variable_scope('cell_1'):
            outputs_1, _ = tf.nn.dynamic_rnn(cell_1, embedded_input,  # ? * 311 * embedding_size
                                             sequence_length=story_length,
                                             initial_state=initial_state_1)

            ##GOOD##
            aa = tf.gather_nd(outputs_1, indices_word)
            #########

            input_rnn2 = []
            length = 0
            zero_padding = tf.zeros([batch_size, max_story_word_length, embedding_size], dtype=aa.dtype)

            for i in xrange(batch_size_int):
                length_padding = max_story_word_length - word_length[i] + 1
                input_rnn2.append(tf.add(zero_padding[i], tf.concat(0, [tf.strided_slice(aa, [length, 0],
                                                                           [length + word_length[i]-1, 100],
                                                                           [1, 1]), tf.zeros([length_padding, embedding_size], dtype=aa.dtype)])))
                length += word_length[i]

            input_rnn2 = tf.pack(input_rnn2)


        with tf.variable_scope('cell_2'):
            outputs_2, last_state_2 = tf.nn.dynamic_rnn(cell_2, input_rnn2,
                                                        sequence_length=word_length,
                                                        initial_state=initial_state_2)

            # TODO find a way to extract the sentence indices
            # bb = tf.gather_nd(outputs_2, indices_sentence)
            # bb = tf.reshape(bb, [batch_size_int, -1, embedding_size]) # wrong: need to change it

        # with tf.variable_scope('cell_3'):
        #     outputs_3, last_state_3 = tf.nn.dynamic_rnn(cell_3, bb,
        #                                                 initial_state=initial_state_3)

        # Output
        # TODO make RNN for Output - "transfer learning from the encoder?"
        output = get_output(last_state_2,
                            vocab_size=vocab_size,
                            initializer=normal_initializer
                            )
        prediction = tf.argmax(output, 1)

        # Training
        loss = get_loss(output, targets, mode)
        train_op = training_optimizer(loss, params, mode)

        if debug:
            # tf.contrib.layers.summarize_tensor(offset, 'offset')
            # tf.contrib.layers.summarize_tensor(word_flattened_indices, 'word_flattened_indices')
            # tf.contrib.layers.summarize_tensor(embedded_input, 'embedded_input')
            # tf.contrib.layers.summarize_tensor(flattened_output_1, 'flattened_output_1')
            # tf.contrib.layers.summarize_tensor(selected_rows_word, 'selected_rows_word')
            # tf.contrib.layers.summarize_tensor(input_rnn2, 'input_rnn2')
            # tf.contrib.layers.summarize_tensor(indices_word, 'indices_word')
            # tf.contrib.layers.summarize_tensor(story_length, 'story_length')
            # tf.contrib.layers.summarize_tensor(last_state_3, 'last_state')
            # tf.contrib.layers.summarize_tensor(output, 'output')
            tf.contrib.layers.summarize_variables()

            tf.add_check_numerics_ops()

        return prediction, loss, train_op


def get_story_word_length(story, token, vocab_size, embedding_size, scope=None):
    """
    Find the word length of a sentence.
    """
    with tf.variable_scope(scope, 'WordLength'):
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
        embedding_mask = tf.constant([1 if i == token else 0 for i in range(vocab_size)],
                                     dtype=tf.float32,
                                     shape=[vocab_size, 1])
        embedding_params_masked = embedding_params * embedding_mask
        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)
        used = tf.sign(tf.reduce_max(tf.abs(story_embedding), reduction_indices=[-1]))
        tmp = tf.cast(tf.reduce_sum(used, reduction_indices=[-1]), tf.int32)
        length = tf.reduce_max(tmp, reduction_indices=[-1])
        return length


def get_story_sentence_length(story, token, vocab_size, embedding_size, scope=None):
    """
    Find the word length of a sentence.
    """
    with tf.variable_scope(scope, 'SentenceLength'):
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
        embedding_mask = tf.constant([1 if i == token else 0 for i in range(vocab_size)],
                                     dtype=tf.float32,
                                     shape=[vocab_size, 1])
        embedding_params_masked = embedding_params * embedding_mask
        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)
        used = tf.sign(tf.reduce_max(tf.abs(story_embedding), reduction_indices=[-1]))
        tmp = tf.cast(tf.reduce_sum(used, reduction_indices=[-1]), tf.int32)
        length = tf.reduce_max(tmp, reduction_indices=[-1])
        return length


def get_story_char_length(sequence, scope=None):
    """
    Find the char length of a sentence that has been padded with zeros.
    """
    with tf.variable_scope(scope, 'SentenceLength'):  # ? * 1 * 311 * 1O
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=[-1]))
        tmp = tf.cast(tf.reduce_sum(used, reduction_indices=[-1]), tf.int32)
        length = tf.reduce_max(tmp, reduction_indices=[-1])
        return length


def get_sentence_indices(story, token_sentence, scope=None):
    with tf.variable_scope(scope, 'SentenceIndices'):
        dots = tf.constant(token_sentence, dtype=tf.int64)
        where = tf.equal(tf.squeeze(story, [1]), dots)
        indices = tf.cast(tf.where(where), tf.int32)
        return indices


def get_word_indices(story, token_space, scope=None):
    with tf.variable_scope(scope, 'WordIndices'):
        spaces = tf.constant(token_space, dtype=tf.int64)
        where = tf.equal(tf.squeeze(story, [1]), spaces)
        indices = tf.cast(tf.where(where), tf.int32)
        return indices


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
