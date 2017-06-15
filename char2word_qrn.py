"""
Implementation of a Query-Reduction Network [arXiv:1606.04582] with TensorFlow
"""

import sys
import re
import tarfile
from functools import reduce
import numpy as np
import os
import datetime

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

from qrn.qrncell import QRNCell


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    def __init__(self):
        pass

    batch_size = 128
    embed_size = 5
    hidden_size_module = 5
    hidden_size_qrn = 50
    max_epochs = 600
    dropout = 0.90
    lr = 0.0035  # the learning rate
    L2 = 0.0002  # the coefficient for L2 regularization

    vocab_char_size = None
    vocab_word_size = None
    num_steps_story_char = None
    num_steps_query_char = None
    num_steps_story_word = None
    num_steps_query_word = None
    num_steps_story = None


class NeuralModel:
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        self.inputs_story, self.inputs_question = self.add_embedding()
        story_question_state = self.add_model(self.inputs_story, self.inputs_question)

        self.output = self.add_projection(story_question_state)

        with tf.name_scope('Accuracy'):
            self.predictions = tf.nn.softmax(self.output)
            self.one_hot_prediction = tf.argmax(self.predictions, 1)
            correct_prediction = tf.equal(tf.argmax(self.labels_placeholder, 1), self.one_hot_prediction)
            self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))

        with tf.name_scope('Loss'):
            loss, lossL2 = self.add_loss_op(self.output)
            self.calculate_loss = loss + self.config.L2 * lossL2
        with tf.name_scope('Train'):
            self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors.
        """
        self.input_story_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps_story_char], name='InputStory')
        self.input_question_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps_query_char], name='InputQuestion')
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.vocab_word_size], name='Target')
        self.X_length = tf.placeholder(tf.int32, shape=[None], name='X_length')
        self.Y_length = tf.placeholder(tf.int32, shape=[None], name='Y_length')
        self.qX_length = tf.placeholder(tf.int32, shape=[None], name='qX_length')
        self.qY_length = tf.placeholder(tf.int32, shape=[None], name='qY_length')
        self.Indices_word = tf.placeholder(tf.int32, shape=[self.config.batch_size, None, 2], name='Indices_word')
        self.qIndices_word = tf.placeholder(tf.int32, shape=[self.config.batch_size, None, 2], name='qIndices_word')
        self.Indices_sentence = tf.placeholder(tf.int32, shape=[self.config.batch_size, None, 2],
                                               name='Indices_sentence')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_embedding(self):
        """Add embedding layer.

        Returns:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        """
        embedding = tf.get_variable('Embedding', [self.config.vocab_char_size, self.config.embed_size],
                                    trainable=True)
        inputs_story = tf.nn.embedding_lookup(embedding, self.input_story_placeholder)
        inputs_question = tf.nn.embedding_lookup(embedding, self.input_question_placeholder)

        return inputs_story, inputs_question

    def add_projection(self, rnn_output):
        """Adds a projection layer.

        The projection layer transforms the hidden representation to a distribution
        over the vocabulary.

        Args:
          rnn_output: List of length num_steps, each of whose elements should be
                       a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each a tensor of shape
                   (batch_size, len(vocab))
        """
        with tf.variable_scope('Projection'):
            U = tf.get_variable('Weights',
                                [self.config.hidden_size_qrn, self.config.vocab_word_size], trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('Bias', [self.config.vocab_word_size])
            outputs = tf.matmul(rnn_output, U) + b

        return outputs

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.

        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
          lossL2: A 0-d tensor (scalar)
        """
        var = tf.trainable_variables()
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.labels_placeholder))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in var if 'Bias' not in v.name])

        return loss, lossL2

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)

        return train_op

    def add_model(self, inputs_story, inputs_question):
        """Creates the QRN model.
        Args:
          inputs_story: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size)
        """
        with tf.variable_scope('Char2Word') as scope:
            # Char2Word
            cell0 = tf.contrib.rnn.BasicRNNCell(self.config.hidden_size_module)
            cell0 = tf.contrib.rnn.DropoutWrapper(cell0, input_keep_prob=self.dropout_placeholder,
                                                  output_keep_prob=self.dropout_placeholder)
            self.initial_state0 = cell0.zero_state(self.config.batch_size, tf.float32)

            self.outputs_story0, self.final_state0 = tf.nn.dynamic_rnn(cell0, inputs_story,
                                                                       sequence_length=self.X_length,
                                                                       initial_state=self.initial_state0)

            scope.reuse_variables()
            self.outputs_question0, _ = tf.nn.dynamic_rnn(cell0, inputs_question,
                                                          sequence_length=self.qX_length,
                                                          initial_state=self.initial_state0)

            indices_word = tf.reshape(self.Indices_word, [-1, 2])
            # extracting the time steps corresponding to the end of words indices_word for the sentences
            inputs1 = tf.gather_nd(self.outputs_story0, indices_word)
            inputs1 = tf.reshape(inputs1, [self.config.batch_size, self.config.num_steps_story_word,
                                           self.config.hidden_size_module])
            qindices_word = tf.reshape(self.qIndices_word, [-1, 2])
            # extracting the time steps corresponding to the end of words indices_word for the questions
            qinputs1 = tf.gather_nd(self.outputs_question0, qindices_word)
            qinputs1 = tf.reshape(qinputs1, [self.config.batch_size, self.config.num_steps_query_word,
                                             self.config.hidden_size_module])

        with tf.variable_scope('Word2Sentence') as scope:
            # Word2Sentence
            cell1 = tf.contrib.rnn.BasicRNNCell(self.config.hidden_size_qrn)
            cell1 = tf.contrib.rnn.DropoutWrapper(cell1, input_keep_prob=self.dropout_placeholder,
                                                  output_keep_prob=self.dropout_placeholder)
            self.initial_state1 = cell1.zero_state(self.config.batch_size, tf.float32)

            self.outputs_story1, self.final_state1 = tf.nn.dynamic_rnn(cell1, inputs1,
                                                                       sequence_length=self.Y_length,
                                                                       initial_state=self.initial_state1)

            scope.reuse_variables()
            _, self.final_question1 = tf.nn.dynamic_rnn(cell1, qinputs1,
                                                        sequence_length=self.qY_length,
                                                        initial_state=self.initial_state1)

            indices_sentence = tf.reshape(self.Indices_sentence, [-1, 2])
            # extracting the time steps corresponding to the end of sentences indices_sentence for the stories
            inputs2 = tf.gather_nd(self.outputs_story1, indices_sentence)
            inputs2 = tf.reshape(inputs2, [self.config.batch_size, self.config.num_steps_story,
                                           self.config.hidden_size_qrn])

            qinputs2 = tf.expand_dims(self.final_question1, 1)
            qinputs2 = tf.tile(qinputs2, tf.stack([1, self.config.num_steps_story, 1]), name=None)

        with tf.variable_scope('QRN') as scope:
            # the qrn cell
            qrn = QRNCell(self.config.hidden_size_qrn, self.config.hidden_size_qrn)

            a, b = custom_bidirectional_dynamic_rnn(qrn, qrn,
                                     [inputs2, qinputs2],
                                     dtype=tf.float32)

            scope.reuse_variables()

            a, b = custom_bidirectional_dynamic_rnn(qrn, qrn, [inputs2, tf.reduce_sum(a, 0)], dtype=tf.float32)
            a, b = custom_bidirectional_dynamic_rnn(qrn, qrn, [inputs2, tf.reduce_sum(a, 0)], dtype=tf.float32)

        return tf.reduce_sum(b, 0)

    def predict(self, session, data):
        # validation
        input_story, input_question, input_labels, \
        X_length, Y_length, Indices_word, Indices_sentence, \
        qX_length, qY_length, qIndices_word = data

        config = self.config
        dp = 1

        n_data = len(input_story)
        batches = zip(range(0, n_data - config.batch_size, config.batch_size),
                      range(config.batch_size, n_data, config.batch_size))
        batches = [(start, end) for start, end in batches]

        test_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        for step, (start, end) in enumerate(batches):
            a = [[y[0] - start, y[1]] for x in Indices_word[start:end] for y in x]
            b = [a[i:i + self.config.num_steps_story_word]
                 for i in range(0, len(a), self.config.num_steps_story_word)]

            c = [[y[0] - start, y[1]] for x in Indices_sentence[start:end] for y in x]
            d = [c[i:i + self.config.num_steps_story]
                 for i in range(0, len(c), self.config.num_steps_story)]

            qa = [[y[0] - start, y[1]] for x in qIndices_word[start:end] for y in x]
            qb = [qa[i:i + self.config.num_steps_query_word]
                  for i in range(0, len(qa), self.config.num_steps_query_word)]

            feed = {self.input_story_placeholder: input_story[start:end],
                    self.input_question_placeholder: input_question[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: dp,
                    self.X_length: X_length[start:end],
                    self.Y_length: Y_length[start:end],
                    self.qX_length: qX_length[start:end],
                    self.qY_length: qY_length[start:end],
                    self.Indices_word: b,
                    self.Indices_sentence: d,
                    self.qIndices_word: qb}
            loss, total_correct = session.run([self.calculate_loss, self.correct_predictions], feed_dict=feed)
            total_processed_examples += end - start
            total_correct_examples += total_correct
            test_loss.append(loss)
        acc = total_correct_examples / float(total_processed_examples)

        return np.mean(test_loss), acc

    def run_epoch(self, session, data, train_op=None, verbose=10):
        # training
        input_story, input_question, input_labels, X_length, Y_length,\
        Indices_word, Indices_sentence, qX_length, qY_length, qIndices_word = data

        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1

        n_data = len(input_story)
        batches = zip(range(0, n_data - config.batch_size, config.batch_size),
                      range(config.batch_size, n_data, config.batch_size))
        batches = [(start, end) for start, end in batches]
        np.random.shuffle(batches)
        n_val = int(len(batches) * 0.1)
        batches_train = batches[:-n_val]
        batches_val = batches[-n_val:]

        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(batches_train)
        for step, (start, end) in enumerate(batches_train):
            a = [[y[0] - start, y[1]] for x in Indices_word[start:end] for y in x]
            b = [a[i:i + self.config.num_steps_story_word]
                 for i in range(0, len(a), self.config.num_steps_story_word)]

            c = [[y[0] - start, y[1]] for x in Indices_sentence[start:end] for y in x]
            d = [c[i:i + self.config.num_steps_story]
                 for i in range(0, len(c), self.config.num_steps_story)]

            qa = [[y[0] - start, y[1]] for x in qIndices_word[start:end] for y in x]
            qb = [qa[i:i + self.config.num_steps_query_word]
                  for i in range(0, len(qa), self.config.num_steps_query_word)]

            feed = {self.input_story_placeholder: input_story[start:end],
                    self.input_question_placeholder: input_question[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: dp,
                    self.X_length: X_length[start:end],
                    self.Y_length: Y_length[start:end],
                    self.qX_length: qX_length[start:end],
                    self.qY_length: qY_length[start:end],
                    self.Indices_word: b,
                    self.Indices_sentence: d,
                    self.qIndices_word: qb}
            loss, total_correct, _ = session.run([self.calculate_loss, self.correct_predictions, train_op],
                                                 feed_dict=feed)
            total_processed_examples += end - start
            total_correct_examples += total_correct
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')
        train_acc = total_correct_examples / float(total_processed_examples)

        total_correct_examples = 0
        total_processed_examples = 0
        for step, (start, end) in enumerate(batches_val):
            a = [[y[0] - start, y[1]] for x in Indices_word[start:end] for y in x]
            b = [a[i:i + self.config.num_steps_story_word]
                 for i in range(0, len(a), self.config.num_steps_story_word)]

            c = [[y[0] - start, y[1]] for x in Indices_sentence[start:end] for y in x]
            d = [c[i:i + self.config.num_steps_story]
                 for i in range(0, len(c), self.config.num_steps_story)]

            qa = [[y[0] - start, y[1]] for x in qIndices_word[start:end] for y in x]
            qb = [qa[i:i + self.config.num_steps_query_word]
                  for i in range(0, len(qa), self.config.num_steps_query_word)]

            feed = {self.input_story_placeholder: input_story[start:end],
                    self.input_question_placeholder: input_question[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: 1,
                    self.X_length: X_length[start:end],
                    self.Y_length: Y_length[start:end],
                    self.qX_length: qX_length[start:end],
                    self.qY_length: qY_length[start:end],
                    self.Indices_word: b,
                    self.Indices_sentence: d,
                    self.qIndices_word: qb}
            total_correct = session.run(self.correct_predictions, feed_dict=feed)
            total_processed_examples += end - start
            total_correct_examples += total_correct

        val_acc = total_correct_examples / float(total_processed_examples)

        return np.mean(total_loss), train_acc, val_acc


def tokenize_word(sent):
    """Return the word tokens of a sentence excluding punctuation.
    >> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', 'Bob', 'went', 'to', 'the', 'kitchen']
    """
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() != '.']


def tokenize_char(sent):
    """
    Return the character tokens of a sentence including punctuation.
    """
    return list(sent.lower())


def parse_stories_char(lines, only_supporting=False):
    """
    Parse the bAbI task format at the character level.
    If only_supporting is True, only the sentences that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize_char(q)
            if only_supporting:
                # only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
                ques = [x for x in q if x]
            else:
                # provide all the substories
                substory = [x for x in story if x]
                ques = [x for x in q if x]
            data.append((substory, ques, a))
            story.append('')
        else:
            sent = tokenize_char(line)
            story.append(sent)
    return data


def get_stories(f):
    """Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories_char(f.readlines())
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story_char), question_char, answer) for story_char, question_char, answer in data]
    return data


def parse_stories_word(lines, only_supporting=False):
    """
    Parse the bAbI task format at the word level.
    If only_supporting is True, only the sentences that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize_word(q)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
                ques = [x for x in q if x]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
                ques = [x for x in q if x]
            data.append((substory, ques, a))
            story.append('')
        else:
            sent = tokenize_word(line)
            story.append(sent)
    return data


def metrics(f):
    """
    Compute some core metrics regarding the processing of the model.
    """
    data = parse_stories_word(f.readlines())
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(story_word, question_word, answer) for story_word, question_word, answer in data]

    story_word_maxlen = max(map(len, (flatten(x) for x, _, _ in data)))
    query_word_maxlen = max(map(len, (x for _, x, _ in data)))
    story_maxsteps = max(map(len, (x for x, _, _ in data)))

    return story_word_maxlen, query_word_maxlen, story_maxsteps


def vectorize_stories(data_char, char_idx, word_idx):
    """
    Vectorize the stories, questions and answers.
    Extract the stories and questions lengths and the words and sentences indices.
    """
    X = []
    Xq = []
    Y = []
    X_length = []
    Y_length = []
    qX_length = []
    qY_length = []
    Indices_word = []
    qIndices_word = []
    Indices_sentence = []
    k = 0

    for story_char, question_char, answer in data_char:
        # story
        x = [char_idx[c] for c in story_char]
        X_length.append(len(x))
        for _ in range(story_char_maxlen - len(x)):
            x.append(0)
        assert len(x) == story_char_maxlen
        indices_word = [[k, i] for i, o in enumerate(x) if o == char_idx[" "] or o == char_idx["."]]
        indices_sentence = [[k, indices_word.index([k, i])] for i, o in enumerate(x) if o == char_idx["."]]
        X.append(x)
        Y_length.append(len(indices_word))
        indices_word += [indices_word[-1]] * (story_word_maxlen - len(indices_word))
        indices_sentence += [indices_sentence[-1]] * (story_maxsteps - len(indices_sentence))
        Indices_word.append(indices_word)
        Indices_sentence.append(indices_sentence)

        # question
        xq = [char_idx[c] for c in question_char]
        qX_length.append(len(xq))
        for _ in range(query_char_maxlen - len(xq)):
            xq.append(0)
        assert len(xq) == query_char_maxlen
        qindices_word = [[k, i] for i, o in enumerate(xq) if o == char_idx[" "] or o == char_idx["?"]]
        Xq.append(xq)
        qY_length.append(len(qindices_word))
        qindices_word += [qindices_word[-1]] * (query_word_maxlen - len(qindices_word))
        qIndices_word.append(qindices_word)

        # answer
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        Y.append(y)

        k += 1

    return X, Xq, Y, X_length, Y_length, Indices_word, Indices_sentence, qX_length, qY_length, qIndices_word


def custom_bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                                     initial_state_fw=None, initial_state_bw=None,
                                     dtype=None, parallel_iterations=None,
                                     swap_memory=False, time_major=False, scope=None):
    """Modified implementation of a bidirectional dynamic rnn suitable for the QRN model.
      Args:
        cell_fw: An instance of RNNCell, to be used for forward direction.
        cell_bw: An instance of RNNCell, to be used for backward direction.
        inputs: The RNN inputs.
          If time_major == False (default), this must be a tensor of shape:
            `[batch_size, max_time, input_size]`.
          If time_major == True, this must be a tensor of shape:
            `[max_time, batch_size, input_size]`.
          [batch_size, input_size].
        sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
          containing the actual lengths for each of the sequences in the batch.
          If not provided, all batch entries are assumed to be full sequences; and
          time reversal is applied from time `0` to `max_time` for each sequence.
        initial_state_fw: (optional) An initial state for the forward RNN.
          This must be a tensor of appropriate type and shape
          `[batch_size, cell_fw.state_size]`.
          If `cell_fw.state_size` is a tuple, this should be a tuple of
          tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
        initial_state_bw: (optional) Same as for `initial_state_fw`, but using
          the corresponding properties of `cell_bw`.
        dtype: (optional) The data type for the initial states and expected output.
          Required if initial_states are not provided or RNN states have a
          heterogeneous dtype.
        parallel_iterations: (Default: 32).  The number of iterations to run in
          parallel.  Those operations which do not have any temporal dependency
          and can be run in parallel, will be.  This parameter trades off
          time for space.  Values >> 1 use more memory but take less time,
          while smaller values use less memory but computations take longer.
        swap_memory: Transparently swap the tensors produced in forward inference
          but needed for back prop from GPU to CPU.  This allows training RNNs
          which would typically not fit on a single GPU, with very minimal (or no)
          performance penalty.
        time_major: The shape format of the `inputs` and `outputs` Tensors.
          If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
          If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
          Using `time_major = True` is a bit more efficient because it avoids
          transposes at the beginning and end of the RNN calculation.  However,
          most TensorFlow data is batch-major, so by default this function
          accepts input and emits output in batch-major form.
        scope: VariableScope for the created subgraph; defaults to
          "bidirectional_rnn"
      Returns:
        A tuple (outputs, output_states) where:
          outputs: A tuple (output_fw, output_bw) containing the forward and
            the backward rnn output `Tensor`.
            If time_major == False (default),
              output_fw will be a `Tensor` shaped:
              `[batch_size, max_time, cell_fw.output_size]`
              and output_bw will be a `Tensor` shaped:
              `[batch_size, max_time, cell_bw.output_size]`.
            If time_major == True,
              output_fw will be a `Tensor` shaped:
              `[max_time, batch_size, cell_fw.output_size]`
              and output_bw will be a `Tensor` shaped:
              `[max_time, batch_size, cell_bw.output_size]`.
            It returns a tuple instead of a single concatenated `Tensor`, unlike
            in the `bidirectional_rnn`. If the concatenated one is preferred,
            the forward and backward outputs can be concatenated as
            `tf.concat(outputs, 2)`.
          output_states: A tuple (output_state_fw, output_state_bw) containing
            the forward and the backward final states of bidirectional rnn.
      Raises:
        TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
      """
    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = tf.nn.dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
                initial_state=initial_state_fw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=fw_scope)

        # Backward direction
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1

        def _reverse(input_, seq_lengths, seq_dim, batch_dim):
            if seq_lengths is not None:
                return array_ops.reverse_sequence(
                    input=input_, seq_lengths=seq_lengths,
                    seq_dim=seq_dim, batch_dim=batch_dim)
            else:
                return array_ops.reverse(input_, axis=[seq_dim])

        with vs.variable_scope("bw") as bw_scope:
            inputs_reverse_story = _reverse(
                inputs[0], seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)
            inputs_reverse_question = _reverse(
                inputs[1], seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)
            tmp, output_state_bw = tf.nn.dynamic_rnn(
                cell=cell_bw, inputs=[inputs_reverse_story, inputs_reverse_question], sequence_length=sequence_length,
                initial_state=initial_state_bw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=bw_scope)

            output_bw = _reverse(
                tmp, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)

    outputs = (output_fw, output_bw)
    output_states = (output_state_fw, output_state_bw)

    return outputs, output_states


tasks = [
    'qa1_single-supporting-fact', 'qa2_two-supporting-facts', 'qa3_three-supporting-facts',
    'qa4_two-arg-relations', 'qa5_three-arg-relations', 'qa6_yes-no-questions', 'qa7_counting',
    'qa8_lists-sets', 'qa9_simple-negation', 'qa10_indefinite-knowledge',
    'qa11_basic-coreference', 'qa12_conjunction', 'qa13_compound-coreference',
    'qa14_time-reasoning', 'qa15_basic-deduction', 'qa16_basic-induction', 'qa17_positional-reasoning',
    'qa18_size-reasoning', 'qa19_path-finding', 'qa20_agents-motivations'
]

if __name__ == "__main__":
    np.random.seed(2930)  # for reproducibility
    verbose = True

    path = 'datasets/babi_tasks_data_1_20_v1.2.tar.gz'
    tar = tarfile.open(path)
    tasks_dir = 'tasks_1-20_v1-2/en-10k/'

    for task in tasks:
        print(task)

        task_path = tasks_dir + task + '_{}.txt'
        train_char = get_stories(tar.extractfile(task_path.format('train')))
        test_char = get_stories(tar.extractfile(task_path.format('test')))

        vocab_char = sorted(reduce(lambda x, y: x | y, (set(story_char + question_char)
                                                        for story_char, question_char, answer in
                                                        train_char + test_char)))
        vocab_word = sorted(set(answer for story_char, question_char, answer in train_char + test_char))

        # reserve 0 for masking via pad_sequences
        vocab_char_size = len(vocab_char) + 1
        vocab_word_size = len(vocab_word) + 1
        char_idx = dict((c, i + 1) for i, c in enumerate(vocab_char))
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab_word))
        idx_char = {v: k for k, v in char_idx.iteritems()}
        idx_char[0] = "_PAD"

        story_char_maxlen = max(map(len, (x for x, _, _ in train_char + test_char)))
        query_char_maxlen = max(map(len, (x for _, x, _ in train_char + test_char)))

        story_word_maxlen, query_word_maxlen, story_maxsteps \
            = max(metrics(tar.extractfile(task_path.format('train'))),
                  metrics(tar.extractfile(task_path.format('test'))))

        X, Xq, Y, X_length, Y_length, Indices_word, \
        Indices_sentence, qX_length, qY_length, qIndices_word = vectorize_stories(train_char, char_idx, word_idx)

        tX, tXq, tY, tX_length, tY_length, tIndices_word, \
        tIndices_sentence, tqX_length, tqY_length, tqIndices_word = vectorize_stories(test_char, char_idx, word_idx)

        if verbose:
            print('vocab_char = {}'.format(vocab_char))
            print('vocab_word = {}'.format(vocab_word))
            print('X.shape = {}'.format(np.array(X).shape))
            print('Y.shape = {}'.format(np.array(Xq).shape))
            print('story_char_maxlen, query_char_maxlen = {}, {}'.format(story_char_maxlen, query_char_maxlen))

        config = Config()
        config.vocab_char_size = vocab_char_size
        config.vocab_word_size = vocab_word_size
        config.num_steps_story_char = story_char_maxlen
        config.num_steps_query_char = query_char_maxlen
        config.num_steps_story_word = story_word_maxlen
        config.num_steps_query_word = query_word_maxlen
        config.num_steps_story = story_maxsteps

        with tf.Graph().as_default() as g:
            model = NeuralModel(config)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            with tf.Session() as session:
                session.run(init)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                model_dir = os.path.join("logs_Char2Word_QRN_10k/", task, str(timestamp + "_e" + str(5)))
                writer = tf.summary.FileWriter(model_dir, graph=g)
                # saver.restore(session, "logs_Char2Word/qa1_single-supporting-fact/good/model")
                for epoch in range(config.max_epochs):
                    if verbose:
                        print('Epoch {}'.format(epoch))

                        print(X)
                        print(Xq)
                        print(Y)
                        print(X_length)
                        print(Y_length)
                        print(Indices_word)
                        print(Indices_sentence)
                        print(qX_length)
                        print(qY_length)
                        print(qIndices_word)

                        train_loss, train_acc, val_acc =\
                            model.run_epoch(session, (X, Xq, Y, X_length, Y_length, Indices_word,Indices_sentence,
                                                                                   qX_length, qY_length, qIndices_word),
                                                                         train_op=model.train_step)

                    if verbose:
                        print('Training loss: {}'.format(train_loss))
                        print('Training acc: {}'.format(train_acc))
                        print('Validation acc: {}'.format(val_acc))

                    if epoch % 20 == 0:
                        test_loss, test_acc = model.predict(session, (tX, tXq, tY, tX_length, tY_length,
                                                                      tIndices_word, tIndices_sentence, tqX_length,
                                                                      tqY_length, tqIndices_word))
                        print('Testing loss: {}'.format(test_loss))
                        print('Testing acc: {}'.format(test_acc))
                    if epoch % 100 == 0:
                        save_path = saver.save(session, os.path.join(model_dir, "model"))
                        print("Model saved in file: %s" % save_path)

                    # save TF summaries
                    tf.summary.scalar("train_loss", train_loss)
                    tf.summary.scalar("train_acc", train_acc)
                    tf.summary.scalar("val_acc", val_acc)
                    tf.summary.scalar("test_acc", test_acc)
                    tf.summary.scalar("test_loss", test_loss)
                    train_loss_S = summary_pb2.Summary.Value(tag="train_loss", simple_value=train_loss.item())
                    train_acc_S = summary_pb2.Summary.Value(tag="train_acc", simple_value=train_acc)
                    val_acc_S = summary_pb2.Summary.Value(tag="val_acc", simple_value=val_acc)
                    test_acc_S = summary_pb2.Summary.Value(tag="test_acc", simple_value=test_acc)
                    test_loss_S = summary_pb2.Summary.Value(tag="test_loss", simple_value=test_loss.item())
                    summary = summary_pb2.Summary(value=[train_loss_S, train_acc_S, val_acc_S, test_acc_S, test_loss_S])
                    writer.add_summary(summary, epoch)
