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
    max_story_length = params['max_story_length']
    token_space = params['token_space']
    token_sentence = params['token_sentence']
    hidden_units = params['hidden_units']
    debug = params['debug']

    story = features['story']  # [? * 1 * 311]
    query = features['query']  # [? * 1 * 4]

    batch_size = tf.shape(story)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    with tf.variable_scope(scope, 'Mango', initializer=normal_initializer):
        # Input
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size)],
                                     dtype=tf.float32,
                                     shape=[vocab_size, 1])
        embedding_params_masked = embedding_params * embedding_mask  # [38 * embedding_size]
        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)  # [? * 1 * 311 * embedding_size]
        query_embedding = tf.nn.embedding_lookup(embedding_params_masked, query)

        indices_word = get_word_indices(story, token_space)  # index(words) # TODO add the dots for the missing words
        indices_sentence = get_sentence_indices(story, token_sentence)  # index(sentences)
        indices_word = tf.concat(0, [indices_word, indices_sentence])  # get all the words

        embedded_input = tf.squeeze(story_embedding, [1])
        encoded_query = tf.squeeze(get_input_encoding(query_embedding, ones_initializer, 'QueryEncoding'), [1])

        # Recurrence
        char_length = get_story_char_length(story_embedding)
        word_length = get_story_word_length(story, token_space, token_sentence, vocab_size, embedding_size)
        sentence_length = get_story_sentence_length(story, token_sentence, vocab_size, embedding_size)

        # TODO remove non-linearities between layers
        cell_1 = tf.nn.rnn_cell.GRUCell(hidden_units)
        cell_2 = tf.nn.rnn_cell.GRUCell(hidden_units)
        cell_3 = tf.nn.rnn_cell.GRUCell(hidden_units)
        initial_state_1 = cell_1.zero_state(batch_size, tf.float32)
        initial_state_2 = cell_2.zero_state(batch_size, tf.float32)
        initial_state_3 = cell_3.zero_state(batch_size, tf.float32)

        with tf.variable_scope('cell_1'):
            outputs_1, _ = tf.nn.dynamic_rnn(cell_1, embedded_input,  # [? * 311 * embedding_size]
                                             sequence_length=char_length,
                                             initial_state=initial_state_1)

            outputs_1_masked = tf.gather_nd(outputs_1, indices_word)

            input_rnn2 = []
            length = 0
            zero_padding = tf.zeros([batch_size, max_story_word_length, embedding_size], dtype=outputs_1_masked.dtype)

            for i in xrange(batch_size_int):
                length_padding = max_story_word_length - word_length[i] + 1
                input_rnn2.append(tf.add(zero_padding[i], tf.concat(0, [tf.strided_slice(outputs_1_masked, [length, 0],
                                                                                         [length + word_length[i] - 1,
                                                                                          100],
                                                                                         [1, 1]),
                                                                        tf.zeros([length_padding, embedding_size],
                                                                                 dtype=outputs_1_masked.dtype)])))
                length += word_length[i]

            input_rnn2 = tf.pack(input_rnn2)

        with tf.variable_scope('cell_2'):
            outputs_2, _ = tf.nn.dynamic_rnn(cell_2, input_rnn2,
                                                        sequence_length=word_length,
                                                        initial_state=initial_state_2)

            indices_sentence = get_aaa(story, token_sentence, indices_word)

            # TODO find a way to extract the sentence indices
            outputs_2_masked = tf.gather_nd(tf.reshape(outputs_2, [-1, embedding_size]), indices_sentence)

            input_rnn3 = []
            length = 0
            zero_padding = tf.zeros([batch_size, max_story_length, embedding_size], dtype=outputs_2_masked.dtype)
            for i in xrange(batch_size_int):
                length_padding = max_story_length - sentence_length[i] + 1
                input_rnn3.append(tf.add(zero_padding[i], tf.concat(0, [tf.strided_slice(outputs_2_masked, [length, 0],
                                                                                         [length + sentence_length[
                                                                                             i] - 1,
                                                                                          100],
                                                                                         [1, 1]),
                                                                        tf.zeros([length_padding, embedding_size],
                                                                                 dtype=outputs_2_masked.dtype)])))
                length += sentence_length[i]

            input_rnn3 = tf.pack(input_rnn3)

        with tf.variable_scope('cell_3'):
            outputs_3, last_state_3 = tf.nn.dynamic_rnn(cell_3, input_rnn3,
                                                        initial_state=initial_state_3)

        # Output
        # TODO make RNN for Output - "transfer learning from the encoder?"
        output = get_output(last_state_3, encoded_query,
                            vocab_size=vocab_size,
                            initializer=normal_initializer
                            )
        prediction = tf.argmax(output, 1)

        # Training
        loss = get_loss(output, targets, mode)
        train_op = training_optimizer(loss, params, mode)

        if debug:
            tf.contrib.layers.summarize_tensor(offset, 'offset')
            tf.contrib.layers.summarize_tensor(word_flattened_indices, 'word_flattened_indices')
            tf.contrib.layers.summarize_tensor(embedded_input, 'embedded_input')
            tf.contrib.layers.summarize_tensor(flattened_output_1, 'flattened_output_1')
            tf.contrib.layers.summarize_tensor(selected_rows_word, 'selected_rows_word')
            tf.contrib.layers.summarize_tensor(input_rnn2, 'input_rnn2')
            tf.contrib.layers.summarize_tensor(indices_word, 'indices_word')
            tf.contrib.layers.summarize_tensor(char_length, 'char_length')
            tf.contrib.layers.summarize_tensor(last_state_3, 'last_state')
            tf.contrib.layers.summarize_tensor(output, 'output')
            tf.contrib.layers.summarize_variables()

            tf.add_check_numerics_ops()

        return prediction, loss, train_op


def get_input_encoding(embedding, initializer=None, scope=None):
    """
    Implementation of a Position Encoding. The mask allows
    the ordering of words in a sentence to affect the encoding.
    """
    with tf.variable_scope(scope, 'Encoding', initializer=initializer):
        _, _, max_sentence_length, _ = embedding.get_shape().as_list()
        positional_mask = tf.get_variable('positional_mask', [max_sentence_length, 1])
        encoded_input = tf.reduce_sum(embedding * positional_mask, reduction_indices=[2])
        return encoded_input


def get_story_word_length(story, token_word, token_sentence, vocab_size, embedding_size, scope=None):
    """
    Find the word length of a sentence.
    """
    with tf.variable_scope(scope, 'WordLength'):
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
        embedding_mask = tf.constant([1 if i == token_word or i == token_sentence else 0 for i in range(vocab_size)],
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


def get_aaa(story, token_sentence, indices_word, scope=None):
    with tf.variable_scope(scope, 'Sentence_Indices'):
        gather = tf.gather_nd(tf.squeeze(story, [1]), indices_word)
        dots = tf.constant(token_sentence, dtype=tf.int64)
        where = tf.equal(gather, dots)
        indices = tf.cast(tf.where(where), tf.int32)
        return indices


def get_word_indices(story, token_space, scope=None):
    with tf.variable_scope(scope, 'WordIndices'):
        spaces = tf.constant(token_space, dtype=tf.int64)
        where = tf.equal(tf.squeeze(story, [1]), spaces)
        indices = tf.cast(tf.where(where), tf.int32)
        return indices


def get_output(last_state, encoded_query, vocab_size, activation=tf.nn.relu, initializer=None, scope=None):
    with tf.variable_scope(scope, 'Output', initializer=initializer):
        _, embedding_size = last_state.get_shape().as_list()

        # Use the encoded_query to attend over memories (hidden states of dynamic last_state cell blocks)
        attention = tf.reduce_sum(last_state * encoded_query, reduction_indices=[1])

        # Subtract max for numerical stability (softmax is shift invariant)
        attention_max = tf.reduce_max(attention, reduction_indices=[-1], keep_dims=True)
        attention = tf.nn.softmax(attention - attention_max)
        attention = tf.expand_dims(attention, 1)

        # Weight memories by attention vectors
        u = last_state * attention

        R = tf.get_variable('R', [embedding_size, vocab_size])
        H = tf.get_variable('H', [embedding_size, embedding_size])

        y = tf.matmul(activation(encoded_query + tf.matmul(u, H)), R)
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
