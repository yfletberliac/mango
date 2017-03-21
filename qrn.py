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

from utils.qrncell import QRNCell


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    batch_size = 32
    embed_size = 100
    hidden_size = 100
    vocab_size = None
    num_steps_story = None
    num_steps_question = None
    max_epochs = 300
    dropout = 1
    lr = 0.001


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
            self.calculate_loss = self.add_loss_op(self.output)
        with tf.name_scope('Train'):
            self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors
        """
        self.input_story_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps_story], name='InputStory')
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
        embedding = tf.get_variable('Embedding', [self.config.vocab_size, self.config.embed_size], trainable=True)
        inputs_story = tf.nn.embedding_lookup(embedding, self.input_story_placeholder)
        inputs_question = tf.nn.embedding_lookup(embedding, self.input_question_placeholder)

        return inputs_story, inputs_question

    def add_projection(self, rnn_output):
        """Adds a projection layer.

        The projection layer transforms the hidden representation to a distribution
        over the vocabulary.

        Args:
          rnn_outputs: List of length num_steps, each of whose elements should be
                       a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each a tensor of shape
                   (batch_size, len(vocab))
        """
        with tf.variable_scope('Projection'):
            U = tf.get_variable('Weights',
                                [self.config.hidden_size, self.config.vocab_size])
            b = tf.get_variable('Bias', [self.config.vocab_size])
            output = tf.matmul(rnn_output, U) + b

        return output

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.

        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
        """
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.labels_placeholder))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))

        return loss

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
        qrn = QRNCell(self.config.hidden_size, self.config.hidden_size)

        with tf.variable_scope('stacked_qrn') as scope:
            a, b = tf.nn.dynamic_rnn(qrn,
                                     [inputs_story, inputs_question],
                                     sequence_length=self.X_length,
                                     dtype=tf.float32)

            scope.reuse_variables()

            a, b = tf.nn.dynamic_rnn(qrn, [inputs_story, a], dtype=tf.float32)
            a, b = tf.nn.dynamic_rnn(qrn, [inputs_story, a], dtype=tf.float32)

        return b

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
    """Return the tokens of a sentence including punctuation.
    >> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', 'Bob', 'went', 'to', 'the', 'kitchen']
    """
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def tokenize_char(sent):
    """
    Tokenize a string by splitting on characters.
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
            substory = None
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
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if
            not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, maxlen_X, maxlen_Xq):
    X = []
    Xq = []
    Y = []
    X_length = []

    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]

        X_length.append(len(x))

        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    return pad_sequences(X, maxlen=maxlen_X), pad_sequences(Xq, maxlen=maxlen_Xq), np.array(Y), X_length


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0):
    '''Pads each sequence to the same length:
    the length of the longest sequence.

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

tasks = [
    'qa1_single-supporting-fact', 'qa2_two-supporting-facts', 'qa3_three-supporting-facts',
    'qa4_two-arg-relations', 'qa5_three-arg-relations', 'qa6_yes-no-questions', 'qa7_counting',
    'qa8_lists-sets', 'qa9_simple-negation', 'qa10_indefinite-knowledge',
    'qa11_basic-coreference', 'qa12_conjunction', 'qa13_compound-coreference',
    'qa14_time-reasoning', 'qa15_basic-deduction', 'qa16_basic-induction', 'qa17_positional-reasoning',
    'qa18_size-reasoning', 'qa19_path-finding', 'qa20_agents-motivations'
]

if __name__ == "__main__":
    np.random.seed(1337)  # for reproducibility
    verbose = True

    path = 'datasets/babi_tasks_data_1_20_v1.2.tar.gz'
    tar = tarfile.open(path)
    tasks_dir = 'tasks_1-20_v1-2/en/'

    for task in tasks:
        print(task)

        task_path = tasks_dir + task + '_{}.txt'
        train = get_stories(tar.extractfile(task_path.format('train')))
        test = get_stories(tar.extractfile(task_path.format('test')))

        vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        story_maxlen = max(map(len, (x for x, _, _ in train + test)))
        query_maxlen = max(map(len, (x for _, x, _ in train + test)))
        idx_word = {v: k for k, v in word_idx.iteritems()}
        idx_word[0] = "_PAD"

        X, Xq, Y, X_length = vectorize_stories(train, word_idx, story_maxlen, story_maxlen)
        tX, tXq, tY, tX_length = vectorize_stories(test, word_idx, story_maxlen, story_maxlen)

        if verbose:
            print('vocab = {}'.format(vocab))
            print('X.shape = {}'.format(X.shape))
            print('Xq.shape = {}'.format(Xq.shape))
            print('Y.shape = {}'.format(Y.shape))
            print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

        config = Config()
        config.vocab_size = vocab_size
        config.num_steps_story = story_maxlen
        config.num_steps_question = story_maxlen

        with tf.Graph().as_default() as g:
            model = RNN_Model(config)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            with tf.Session() as session:
                session.run(init)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                model_dir = os.path.join("logs_QRN/", task, str(timestamp))
                writer = tf.summary.FileWriter(model_dir, graph=g)
                # saver.restore(session, "logs_Char2Word/qa1_single-supporting-fact/good/model")
                for epoch in range(config.max_epochs):
                    if verbose:
                        print('Epoch {}'.format(epoch))

                    train_loss, train_acc, val_acc, Story, Question, Answer, Prediction = model.run_epoch(session, (X, Xq, Y, X_length), train_op=model.train_step)

                    # save TF summaries
                    tf.summary.scalar("train_loss", train_loss)
                    tf.summary.scalar("train_acc", train_acc)
                    tf.summary.scalar("val_acc", val_acc)
                    train_loss_S = summary_pb2.Summary.Value(tag="train_loss", simple_value=train_loss.item())
                    train_acc_S = summary_pb2.Summary.Value(tag="train_acc", simple_value=train_acc)
                    val_acc_S = summary_pb2.Summary.Value(tag="val_acc", simple_value=val_acc)
                    summary = summary_pb2.Summary(value=[train_loss_S, train_acc_S, val_acc_S])
                    writer.add_summary(summary, epoch)

                    if verbose:
                        print('Training loss: {}'.format(train_loss))
                        print('Training acc: {}'.format(train_acc))
                        print('Validation acc: {}'.format(val_acc))
                        print([idx_word[i] for i in Story[0][30]])
                        print([idx_word[i] for i in Question[0][30]])
                        print('Answer: {}'.format(idx_word[Answer[0][30].tolist().index(1.)]))
                        print('Prediction: {}'.format(idx_word[Prediction[0][30]]))
                    if epoch % 20 == 0:
                        save_path = saver.save(session, os.path.join(model_dir, "model"))
                        print("Model saved in file: %s" % save_path)

                test_acc = model.predict(session, (tX, tXq, tY, tX_length))
                print('Testing acc: {}'.format(test_acc))
