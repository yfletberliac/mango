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
    vocab_size_char = params['vocab_size_char']
    vocab_size_word = params['vocab_size_word']
    max_story_word_length = params['max_story_word_length']
    token_space = params['token_space']
    token_sentence = params['token_sentence']
    hidden_size = params['hidden_size']
    debug = params['debug']

    story = features['story_char']  # [? * 1 * max_story_char_length]

    batch_size = tf.shape(story)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)

    with tf.variable_scope(scope, 'Mango', initializer=normal_initializer):
        ## INPUT embedding
        with tf.variable_scope('Input'):
            embedding_params_story = tf.get_variable('embedding_params_story', [vocab_size_char, embedding_size])
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size_char)],
                                         dtype=tf.float32,
                                         shape=[vocab_size_char, 1])
            embedding_params_masked_story = embedding_params_story * embedding_mask  # [vocab_size * embedding_size]
            story_embedding = tf.nn.embedding_lookup(embedding_params_masked_story, story)  # [? * 1 *
            # max_story_char_length * embedding_size]
            embedded_input = tf.squeeze(story_embedding, [1])  # [? * max_story_char_length * embedding_size]

            ## Get the word and sentence indices
            indices_word = get_space_indices(story, token_space)
            indices_sentence = get_dot_indices(story, token_sentence)  # get all the sentences
            indices_word = tf.concat([indices_word, indices_sentence], 0)  # get all the words

            ## Get the number of characters and words for each story
            char_length = get_story_char_length(story_embedding)
            word_length = get_story_word_length(story, token_space, token_sentence, vocab_size_char, embedding_size)

        ## RECURRENCE - 1st layer
        with tf.variable_scope('Char2Word'):
            # TODO remove non-linearities between layers
            ## Define the cell
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, use_peepholes=True, activation=tf.tanh)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.5)  # doesn't help
            # cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)  # doesn't help
            ## Initial states of the cells
            initial_state = cell.zero_state(batch_size, tf.float32)

            ## Run 1st layer iterations
            outputs_1, _ = tf.nn.dynamic_rnn(cell, embedded_input,
                                             sequence_length=char_length,
                                             initial_state=initial_state)

            outputs_1 = tf.concat(outputs_1, 2)
            ## Extract needed time steps (corresponding to the end of words [indices_word])
            outputs_1_masked = tf.gather_nd(outputs_1, indices_word)  # [len(indices_word) * embedding_size]

            ## Reshape back to [? * max_story_word_length * embedding_size]
            input_rnn2 = []
            length = 0
            zero_padding = tf.zeros([batch_size, max_story_word_length, hidden_size], dtype=outputs_1_masked.dtype)

            for i in xrange(batch_size_int):
                length_padding = max_story_word_length - word_length[i] + 1
                input_rnn2.append(tf.add(zero_padding[i], tf.concat([tf.strided_slice(outputs_1_masked, [length, 0],
                                                                                      [length + word_length[i] - 1,
                                                                                       hidden_size],
                                                                                      [1, 1]),
                                                                     tf.zeros([length_padding, hidden_size],
                                                                              dtype=outputs_1_masked.dtype)], 0)))
                length += word_length[i]

            input_rnn2 = tf.stack(input_rnn2)

        ## RECURRENCE - 2nd layer
        with tf.variable_scope('Word2Word'):
            # TODO remove non-linearities between layers
            ## Define the cell
            cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, use_peepholes=True, activation=tf.tanh)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.5)  # doesn't help
            # cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)  # doesn't help
            ## Initial states of the cells
            initial_state = cell.zero_state(batch_size, tf.float32)

            outputs_2, _ = tf.nn.dynamic_rnn(cell, input_rnn2,
                                             sequence_length=word_length,
                                             initial_state=initial_state)

        # OUTPUT
        # TODO make RNN for Output - "transfer learning from the encoder?"
        output, prediction = get_output(outputs_2,
                                        vocab_size=vocab_size_word,
                                        batch_size=batch_size_int,
                                        initializer=normal_initializer
                                        )
        ## LOSS
        with tf.variable_scope('Loss'):
            loss = get_loss(output, targets, vocab_size_word, word_length, mode)
            ## OPTIMIZATION
            train_op = training_optimizer(loss, params, mode)

        if debug:
            tf.contrib.layers.summarize_tensor(embedded_input, 'embedded_input')

            tf.contrib.layers.summarize_tensor(outputs_1, 'outputs_1')
            tf.contrib.layers.summarize_tensor(outputs_2, 'outputs_2')

            tf.contrib.layers.summarize_tensor(char_length, 'char_length')
            tf.contrib.layers.summarize_tensor(word_length, 'word_length')

            tf.contrib.layers.summarize_tensor(output, 'output')
            tf.contrib.layers.summarize_tensor(targets, 'targets')

            tf.add_check_numerics_ops()

        return prediction, loss, train_op


def get_story_word_length(story, token_word, token_sentence, vocab_size, embedding_size, scope=None):
    """
    Find the word length of a story.
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


def get_story_char_length(sequence, scope=None):
    """
    Find the char length of a story.
    """
    with tf.variable_scope(scope, 'CharLength'):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=[-1]))
        tmp = tf.cast(tf.reduce_sum(used, reduction_indices=[-1]), tf.int32)
        length = tf.reduce_max(tmp, reduction_indices=[-1])
        return length


def get_dot_indices(story, token_sentence, scope=None):
    """
    Find the 'dot' indices in a story.
    """
    with tf.variable_scope(scope, 'DotIndices'):
        dots = tf.constant(token_sentence, dtype=tf.int64)
        where = tf.equal(tf.squeeze(story, [1]), dots)
        indices = tf.cast(tf.where(where), tf.int32)
        return indices


def get_space_indices(story, token_space, scope=None):
    """
    Find the 'space' indices in a story.
    """
    with tf.variable_scope(scope, 'SpaceIndices'):
        spaces = tf.constant(token_space, dtype=tf.int64)
        where = tf.equal(tf.squeeze(story, [1]), spaces)
        indices = tf.cast(tf.where(where), tf.int32)
        return indices


def get_output(output, vocab_size, batch_size, activation=tf.nn.relu, initializer=None, scope=None):
    """
    Output module.
    """
    with tf.variable_scope(scope, 'Output', initializer=initializer):
        _, _, embedding_size = output.get_shape().as_list()

        R = tf.get_variable('R', [batch_size, embedding_size, vocab_size])
        H = tf.get_variable('H', [batch_size, embedding_size, embedding_size])

        y = tf.matmul(activation(tf.matmul(output, H)), R)
        return y, tf.argmax(y, 2)


def get_loss(output, labels, vocab_size, word_length, mode):
    """
    Function to compute the loss.
    """
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return None
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(labels.get_shape().ndims)
    #     print(output.get_shape().ndims)

    output = tf.reshape(output, [-1, vocab_size])
    labels_flat = tf.reshape(labels, [-1])

    losses = tf.losses.sparse_softmax_cross_entropy(logits=output, labels=labels_flat)

    # Mask the losses
    mask = tf.sign(tf.to_float(labels_flat))
    masked_losses = mask * losses

    # Bring back to [B, T] shape
    masked_losses = tf.reshape(masked_losses, tf.shape(labels))

    # Calculate mean loss
    mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.to_float(word_length)
    mean_loss = tf.reduce_mean(mean_loss_by_example)

    return mean_loss


def training_optimizer(loss, params, mode):
    """
    Optimization module.
    """
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
