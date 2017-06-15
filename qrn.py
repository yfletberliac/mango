"""
Basic RNN for bAbI tasks using Tensorflow
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

    batch_size = 32
    embed_size = 50
    hidden_size = 50
    max_epochs = 5000
    dropout = 0.7
    lr = 0.8
    L2 = 0.001

    vocab_size = None
    num_steps_sentence = None
    num_steps_story = None
    num_steps_question = None


class RNN_Model:
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
        """Generate placeholder variables to represent the input tensors
        """
        self.input_story_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps_story, self.config.num_steps_sentence], name='InputStory')
        self.input_question_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps_question], name='InputQuestion')
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.vocab_size], name='Target')
        self.X_length = tf.placeholder(tf.int32, shape=[None], name='X_length')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_embedding(self):
        """Add embedding layer.

        Returns:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        """
        embedding = tf.get_variable('Embedding', [self.config.vocab_size, self.config.embed_size], trainable=True,
                                    initializer=tf.contrib.layers.xavier_initializer())
        inputs_story = tf.nn.embedding_lookup(embedding, self.input_story_placeholder)
        inputs_question = tf.nn.embedding_lookup(embedding, self.input_question_placeholder)

        # Position Encoding
        inputs_question = tf.expand_dims(inputs_question, 1)
        encoded_story = self.get_position_encoding(inputs_story, self.config.num_steps_sentence, 'StoryEncoding')
        encoded_query = self.get_position_encoding(inputs_question, self.config.num_steps_question, 'QueryEncoding')
        encoded_query = tf.tile(encoded_query, tf.stack([1, self.config.num_steps_story, 1]), name=None)

        return encoded_story, encoded_query

    def get_position_encoding(self, embedding, max_length, scope=None):
        """
        Module described in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852) as Position Encoding (PE).
        The mask allows the ordering of words in a sentence to affect the encoding.
        """
        J, d = max_length, self.config.embed_size
        l = np.zeros((J, d))
        with tf.variable_scope(scope, 'PE'):
            for j in range(J):
                for k in range(d):
                    l[j, k] = (1. - (j + 1.) / J) - ((k + 1.) / d) * (1. - 2. * (j + 1.) / J)
            self.l = tf.constant(l, shape=[J, d])
            m = tf.reduce_sum(embedding * tf.cast(self.l, tf.float32), 2, name='m')
        return m

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
                                [self.config.hidden_size, self.config.vocab_size], trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('Bias', [self.config.vocab_size])
            outputs = tf.matmul(rnn_output, U) + b

        return outputs

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.

        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
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
        optimizer = tf.train.AdagradOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)

        return train_op

    def add_model(self, inputs_story, inputs_question):
        qrn = QRNCell(self.config.hidden_size, self.config.hidden_size)
        qrn = tf.contrib.rnn.DropoutWrapper(qrn, input_keep_prob=self.dropout_placeholder,
                                            output_keep_prob=self.dropout_placeholder)

        with tf.variable_scope('stacked_qrn') as scope:
            a, b = custom_bidirectional_dynamic_rnn(qrn, qrn,
                                                    [inputs_story, inputs_question],
                                                    sequence_length=self.X_length,
                                                    dtype=tf.float32)
            scope.reuse_variables()

            a, b = custom_bidirectional_dynamic_rnn(qrn, qrn, [inputs_story, tf.reduce_sum(a, 0)], dtype=tf.float32)
            a, b = custom_bidirectional_dynamic_rnn(qrn, qrn, [inputs_story, tf.reduce_sum(a, 0)], dtype=tf.float32)

        return tf.reduce_sum(b, 0)

    def predict(self, session, data):
        input_story, input_question, input_labels, X_length = data
        config = self.config
        dp = 1

        n_data = len(input_story)
        batches = zip(range(0, n_data - config.batch_size, config.batch_size),
                      range(config.batch_size, n_data, config.batch_size))
        batches = [(start, end) for start, end in batches]
        total_correct_examples = 0
        total_processed_examples = 0
        for step, (start, end) in enumerate(batches):
            feed = {self.input_story_placeholder: input_story[start:end],
                    self.input_question_placeholder: input_question[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: dp,
                    self.X_length: X_length[start:end]}
            total_correct = session.run(self.correct_predictions, feed_dict=feed)
            total_processed_examples += end - start
            total_correct_examples += total_correct
        acc = total_correct_examples / float(total_processed_examples)

        return acc

    def run_epoch(self, session, data, train_op=None, verbose=10):
        input_story, input_question, input_labels, X_length = data
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
            feed = {self.input_story_placeholder: input_story[start:end],
                    self.input_question_placeholder: input_question[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: dp,
                    self.X_length: X_length[start:end]}
            loss, total_correct, _ = session.run(
                [self.calculate_loss, self.correct_predictions, train_op],
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

        Story = []
        Question = []
        Answer = []
        Prediction = []
        total_correct_examples = 0
        total_processed_examples = 0
        for step, (start, end) in enumerate(batches_val):
            feed = {self.input_story_placeholder: input_story[start:end],
                    self.input_question_placeholder: input_question[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: 1,
                    self.X_length: X_length[start:end]}
            total_correct, prediction = session.run([self.correct_predictions, self.one_hot_prediction], feed_dict=feed)
            total_processed_examples += end - start
            total_correct_examples += total_correct

            Story.append(input_story[start:end])
            Question.append(input_question[start:end])
            Answer.append(input_labels[start:end])
            Prediction.append(prediction)

        val_acc = total_correct_examples / float(total_processed_examples)

        return np.mean(total_loss), train_acc, val_acc, Story, Question, Answer, Prediction


def tokenize_word(sent):
    """Return the tokens of a sentence excluding punctuation.
    >> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', 'Bob', 'went', 'to', 'the', 'kitchen']
    """
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def tokenize_char(sent):
    """
    Return the character tokens of a sentence including punctuation.
    """
    return list(sent.lower())


def parse_stories(lines, only_supporting=False):
    """
    Parse the bAbI task format.
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
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize_word(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    """Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    data = [(story, q, answer) for story, q, answer in data if
            not max_length or len(story) < max_length]
    return data


def vectorize_stories(data, word_idx, sentence_maxlen, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    X_length = []

    for story, query, answer in data:
        sentences = []
        for s in story:
            sentence = [word_idx[w] for w in s]
            for _ in range(sentence_maxlen - len(sentence)):
                sentence.append(0)
            assert len(sentence) == sentence_maxlen
            sentences.append(sentence)
        X_length.append(len(sentences))

        ## story
        for _ in range(story_maxlen - len(sentences)):
            sentences.append([0 for _ in range(sentence_maxlen)])

        ## query
        xq = [word_idx[w] for w in query]
        for _ in range(query_maxlen - len(xq)):
            xq.append(0)
        ## answer
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1

        X.append(sentences)
        Xq.append(xq)
        Y.append(y)

    return X, Xq, np.array(Y), X_length


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
    verbose = True

    path = 'datasets/babi_tasks_data_1_20_v1.2.tar.gz'
    tar = tarfile.open(path)
    tasks_dir = 'tasks_1-20_v1-2/en/'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_1k_final200")

    for task in tasks:
        print(task)
        for seed in [3242, 3892]:
            print("%i (seed)" % seed)
            np.random.seed(seed)  # for reproducibility

            task_path = tasks_dir + task + '_{}.txt'
            train = get_stories(tar.extractfile(task_path.format('train')))
            test = get_stories(tar.extractfile(task_path.format('test')))

            flatten = lambda data: reduce(lambda x, y: x + y, data)
            vocab = sorted(reduce(lambda x, y: x | y, (set(flatten(story) + q + [answer])
                                                       for story, q, answer in train + test)))

            # Reserve 0 for masking via pad_sequences
            vocab_size = len(vocab) + 1
            word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

            sentence_maxlen = max(flatten([[len(s) for s in x] for x, _, _ in train + test]))
            story_maxlen = max(map(len, (x for x, _, _ in train + test)))
            query_maxlen = max(map(len, (x for _, x, _ in train + test)))
            idx_word = {v: k for k, v in word_idx.iteritems()}
            idx_word[0] = "_PAD"

            X, Xq, Y, X_length = vectorize_stories(train, word_idx, sentence_maxlen, story_maxlen, query_maxlen)
            tX, tXq, tY, tX_length = vectorize_stories(test, word_idx, sentence_maxlen, story_maxlen, query_maxlen)

            if verbose:
                print('vocab = {}'.format(vocab))
                print('X.shape = {}'.format(np.array(X).shape))
                print('Xq.shape = {}'.format(np.array(Xq).shape))
                print('Y.shape = {}'.format(Y.shape))
                print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

            config = Config()
            config.vocab_size = vocab_size
            config.num_steps_sentence = sentence_maxlen
            config.num_steps_story = story_maxlen
            config.num_steps_question = query_maxlen

            with tf.Graph().as_default() as g:
                model = RNN_Model(config)
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()

                with tf.Session() as session:
                    session.run(init)
                    model_dir = os.path.join("logs_QRN/", task, str(timestamp+"_"+str(seed)))
                    writer = tf.summary.FileWriter(model_dir, graph=g)
                    # saver.restore(session, "logs_Char2Word/qa1_single-supporting-fact/good/model")
                    for epoch in range(config.max_epochs):
                        print('Epoch {}'.format(epoch))

                        train_loss, train_acc, val_acc, Story, Question, Answer, Prediction = model.run_epoch(session, (
                            X, Xq, Y, X_length), train_op=model.train_step)

                        if verbose:
                            print('Training loss: {}'.format(train_loss))
                            print('Training acc: {}'.format(train_acc))
                            print('Validation acc: {}'.format(val_acc))
                            # print([[idx_word[j] for j in i] for i in Story[0][20]])
                            # print([idx_word[i] for i in Question[0][20]])
                            # print('Answer: {}'.format(idx_word[Answer[0][20].tolist().index(1.)]))
                            # print('Prediction: {}'.format(idx_word[Prediction[0][20]]))

                        if epoch % 20 == 0:
                            test_acc = model.predict(session, (tX, tXq, tY, tX_length))
                            print('Testing acc: {}'.format(test_acc))
                        if epoch % 100 == 0:
                            save_path = saver.save(session, os.path.join(model_dir, "model"))
                            print("Model saved in file: %s" % save_path)

                        # save TF summaries
                        tf.summary.scalar("train_loss", train_loss)
                        tf.summary.scalar("train_acc", train_acc)
                        tf.summary.scalar("val_acc", val_acc)
                        tf.summary.scalar("test_acc", test_acc)
                        train_loss_S = summary_pb2.Summary.Value(tag="train_loss", simple_value=train_loss.item())
                        train_acc_S = summary_pb2.Summary.Value(tag="train_acc", simple_value=train_acc)
                        val_acc_S = summary_pb2.Summary.Value(tag="val_acc", simple_value=val_acc)
                        test_acc_S = summary_pb2.Summary.Value(tag="test_acc", simple_value=test_acc)
                        summary = summary_pb2.Summary(value=[train_loss_S, train_acc_S, val_acc_S, test_acc_S])
                        writer.add_summary(summary, epoch)
