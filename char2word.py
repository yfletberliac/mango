"""
Basic RNN for bAbI tasks using TensorFlow
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


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    def __init__(self):
        pass

    batch_size = 32
    embed_size = 100
    hidden_size = 100
    vocab_char_size = None
    vocab_word_size = None
    num_steps_story_char = None
    num_steps_story_word = None
    max_epochs = 100
    dropout = 1
    lr = 0.002


class NeuralModel:
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        self.inputs_story = self.add_embedding()
        story_output = self.add_model(self.inputs_story)

        self.output = self.add_projection(story_output)

        with tf.name_scope('Accuracy'):
            self.predictions = tf.nn.softmax(self.output)
            one_hot_prediction = tf.argmax(self.predictions, 2)

            mask = tf.sign(tf.to_float(self.labels_placeholder))
            self.masked_one_hot_prediction = tf.cast(mask, 'int32') * tf.cast(one_hot_prediction, 'int32')

            self.correct_prediction = tf.equal(self.labels_placeholder, self.masked_one_hot_prediction)
            self.correct_predictions = tf.reduce_sum(tf.cast(self.correct_prediction, 'int32'))

        self.pred = one_hot_prediction

        with tf.name_scope('Loss'):
            self.calculate_loss = self.add_loss_op(self.output)
        with tf.name_scope('Train'):
            self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors
        """
        self.input_story_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps_story_char], name='InputStory')
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps_story_word], name='Target')
        self.X_length = tf.placeholder(tf.int32, shape=[None], name='X_length')
        self.Y_length = tf.placeholder(tf.int32, shape=[None], name='Y_length')
        self.Indices_word = tf.placeholder(tf.int32, shape=[self.config.batch_size, None, 2], name='Indices_word')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')

    def add_embedding(self):
        """Add embedding layer.

        Returns:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        """
        embedding = tf.get_variable('Embedding', [self.config.vocab_char_size, self.config.embed_size], trainable=True)
        inputs_story = tf.nn.embedding_lookup(embedding, self.input_story_placeholder)

        return inputs_story

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
                                [self.config.batch_size, self.config.hidden_size, self.config.vocab_word_size])
            b = tf.get_variable('Bias', [self.config.vocab_word_size])
            output = tf.matmul(rnn_output, U) + b

        return output

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.

        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.labels_placeholder)

        # mask the losses
        mask = tf.sign(tf.to_float(self.labels_placeholder))
        masked_losses = mask * cross_entropy

        # bring back to [B, T] shape
        masked_losses = tf.reshape(masked_losses, tf.shape(self.labels_placeholder))

        # calculate mean loss
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.to_float(self.Y_length)
        mean_loss = tf.reduce_mean(mean_loss_by_example)

        tf.add_to_collection('total_loss', mean_loss)
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

    def add_model(self, inputs_story):
        """Creates the RNN LM model.
        Args:
          inputs_story: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size)
        """
        with tf.variable_scope('Char2Word'):
            cell0 = tf.contrib.rnn.BasicRNNCell(self.config.hidden_size)
            cell0 = tf.contrib.rnn.DropoutWrapper(cell0, input_keep_prob=self.dropout_placeholder,
                                                  output_keep_prob=self.dropout_placeholder)
            self.initial_state0 = cell0.zero_state(self.config.batch_size, tf.float32)

            self.outputs0, self.final_state0 = tf.nn.dynamic_rnn(cell0, inputs_story,
                                                                 sequence_length=self.X_length,
                                                                 initial_state=self.initial_state0)

            indices_word = tf.reshape(self.Indices_word, [-1, 2])
            # extract the time steps corresponding to the end of words indices_word
            inputs1 = tf.gather_nd(self.outputs0, indices_word)
            print(inputs1)
            inputs1 = tf.reshape(inputs1, [self.config.batch_size, self.config.num_steps_story_word,
                                           self.config.hidden_size])

        with tf.variable_scope('Word2Word'):
            cell1 = tf.contrib.rnn.BasicRNNCell(self.config.hidden_size)
            cell1 = tf.contrib.rnn.DropoutWrapper(cell1, input_keep_prob=self.dropout_placeholder,
                                                  output_keep_prob=self.dropout_placeholder)
            self.initial_state1 = cell1.zero_state(self.config.batch_size, tf.float32)

            self.outputs1, self.final_state1 = tf.nn.dynamic_rnn(cell1, inputs1,
                                                                 sequence_length=self.Y_length,
                                                                 initial_state=self.initial_state1)

        return self.outputs1

    def predict(self, session, data):
        input_story, input_labels, X_length, Y_length, Indices_word, Indices_sentence = data
        config = self.config
        dp = 1

        n_data = len(input_story)
        batches = zip(range(0, n_data - config.batch_size, config.batch_size),
                      range(config.batch_size, n_data, config.batch_size))
        batches = [(start, end) for start, end in batches]
        total_correct_examples = 0
        total_processed_examples = 0
        for step, (start, end) in enumerate(batches):
            a = [[y[0] - start, y[1]] for x in Indices_word[start:end] for y in x]
            b = [a[i:i + story_word_maxlen] for i in range(0, len(a), story_word_maxlen)]

            feed = {self.input_story_placeholder: input_story[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: dp,
                    self.X_length: X_length[start:end],
                    self.Y_length: Y_length[start:end],
                    self.Indices_word: b}
            total_correct = session.run(self.correct_predictions, feed_dict=feed)
            total_processed_examples += sum(Y_length[start:end])
            total_correct_examples += total_correct - ((end - start) * story_word_maxlen - sum(Y_length[start:end]))
        acc = total_correct_examples / float(total_processed_examples)

        return acc

    def run_epoch(self, session, data, train_op=None, verbose=10):
        input_story, input_labels, X_length, Y_length, Indices_word, Indices_sentence = data

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

        total_loss_train = []
        total_loss_val = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(batches_train)
        for step, (start, end) in enumerate(batches_train):
            a = [[y[0] - start, y[1]] for x in Indices_word[start:end] for y in x]
            b = [a[i:i + story_word_maxlen] for i in range(0, len(a), story_word_maxlen)]

            feed = {self.input_story_placeholder: input_story[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: dp,
                    self.X_length: X_length[start:end],
                    self.Y_length: Y_length[start:end],
                    self.Indices_word: b}
            loss_train, total_correct, state0, state1, _ = session.run(
                [self.calculate_loss, self.correct_predictions, self.final_state0, self.final_state1, train_op],
                feed_dict=feed)
            total_processed_examples += sum(Y_length[start:end])
            total_correct_examples += total_correct - ((end - start) * story_word_maxlen - sum(Y_length[start:end]))
            total_loss_train.append(loss_train)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : train_loss = {}'.format(
                    step, total_steps, np.mean(total_loss_train)))
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')
        train_acc = total_correct_examples / float(total_processed_examples)

        Prediction = []
        Mask = []
        Correct = []
        Labels = []
        total_correct_examples = 0
        total_processed_examples = 0
        for step, (start, end) in enumerate(batches_val):
            a = [[y[0] - start, y[1]] for x in Indices_word[start:end] for y in x]
            b = [a[i:i + story_word_maxlen] for i in range(0, len(a), story_word_maxlen)]

            feed = {self.input_story_placeholder: input_story[start:end],
                    self.labels_placeholder: input_labels[start:end],
                    self.dropout_placeholder: 1,
                    self.X_length: X_length[start:end],
                    self.Y_length: Y_length[start:end],
                    self.Indices_word: b}
            loss_val, total_correct, prediction, mask, correct = session.run(
                [self.calculate_loss, self.correct_predictions,
                 self.pred, self.masked_one_hot_prediction, self.correct_prediction], feed_dict=feed)
            total_processed_examples += sum(Y_length[start:end])
            total_correct_examples += total_correct - ((end - start) * story_word_maxlen - sum(Y_length[start:end]))
            total_loss_val.append(loss_val)
            Prediction.append(prediction)
            Mask.append(mask)
            Correct.append(correct)
            Labels.append(input_labels[start:end])
        val_acc = total_correct_examples / float(total_processed_examples)

        return np.mean(total_loss_train), np.mean(total_loss_val), train_acc, val_acc, Prediction, Mask, Correct, Labels


def tokenize_word(sent):
    """Return the tokens of a sentence including punctuation.
    >> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', 'Bob', 'went', 'to', 'the', 'kitchen']
    """
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() != '.']


def tokenize_char(sent):
    """
    Tokenize a string by splitting on characters.
    """
    return list(sent.lower())


def parse_stories(lines):
    """
    Parse the bAbI task format.
    If only_supporting is True, only the sentences that support the answer are kept.
    """
    stories = []
    story_char = []
    story_word = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story_char = []
            story_word = []
        if '\t' in line:
            substory_char = [x for x in story_char if x]
            substory_word = [x for x in story_word if x]
            stories.append((substory_char, substory_word))
            story_char.append('')
            story_word.append('')
        else:
            sentence_char = tokenize_char(line)
            sentence_word = tokenize_word(line)
            story_char.append(sentence_char)
            story_word.append(sentence_word)
    return stories


def get_stories(f):
    """Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories(f.readlines())
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story_char), flatten(story_word)) for story_char, story_word in data]
    return data


def vectorize_stories(data, char_idx, word_idx):
    X = []
    Y = []
    X_length = []
    Y_length = []
    Indices_word = []
    Indices_sentence = []
    k = 0

    for story_char, story_word in data:
        x = [char_idx[c] for c in story_char]
        y = [word_idx[w] for w in story_word]
        indices_word = [[k, i] for i, o in enumerate(x) if o == char_idx[" "] or o == char_idx["."]]
        indices_sentence = [[k, i] for i, o in enumerate(x) if o == char_idx["."]]

        X.append(x)
        Y.append(y)
        X_length.append(len(x))
        Y_length.append(len(y))
        indices_word += [indices_word[-1]] * (story_word_maxlen - len(indices_word))
        Indices_word.append(indices_word)
        Indices_sentence.append(indices_sentence)

        k += 1

    return pad_sequences(X, position='input'), pad_sequences(Y), X_length, Y_length, Indices_word, Indices_sentence


def pad_sequences(sequences, maxlen=None, dtype='int32', position=None,
                  padding='post', truncating='post', value=0):
    """Pads each sequence to the same length:
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
    """
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

    if position == 'input':
        x = (np.ones((nb_samples, story_char_maxlen) + sample_shape) * value).astype(dtype)
    else:
        x = (np.ones((nb_samples, story_word_maxlen) + sample_shape) * value).astype(dtype)

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

    path = 'babi/babi_tasks_data_1_20_v1.2.tar.gz'
    tar = tarfile.open(path)
    tasks_dir = 'tasks_1-20_v1-2/en/'

    for task in tasks:
        print(task)

        task_path = tasks_dir + task + '_{}.txt'
        train = get_stories(tar.extractfile(task_path.format('train')))
        test = get_stories(tar.extractfile(task_path.format('test')))

        vocab_char = sorted(reduce(lambda x, y: x | y, (set(story_char) for story_char, story_word in train + test)))
        vocab_word = sorted(reduce(lambda x, y: x | y, (set(story_word) for story_char, story_word in train + test)))

        # Reserve 0 for masking via pad_sequences
        vocab_char_size = len(vocab_char) + 1
        vocab_word_size = len(vocab_word) + 1
        char_idx = dict((c, i + 1) for i, c in enumerate(vocab_char))
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab_word))
        idx_char = {v: k for k, v in char_idx.iteritems()}
        idx_char[0] = "_PAD"
        idx_word = {v: k for k, v in word_idx.iteritems()}
        idx_word[0] = "_PAD"

        story_char_maxlen = max(map(len, (x for x, _ in train + test)))
        story_word_maxlen = max(map(len, (x for _, x in train + test)))

        X, Y, X_length, Y_length, Indices_word, Indices_sentence = vectorize_stories(train, char_idx, word_idx)
        tX, tY, tX_length, tY_length, tIndices_word, tIndices_sentence = vectorize_stories(test, char_idx, word_idx)

        if verbose:
            print('vocab_char = {}'.format(vocab_char))
            print('vocab_word = {}'.format(vocab_word))
            print('X.shape = {}'.format(X.shape))
            print('Y.shape = {}'.format(Y.shape))
            print('story_char_maxlen, story_word_maxlen = {}, {}'.format(story_char_maxlen, story_word_maxlen))

        config = Config()
        config.vocab_char_size = vocab_char_size
        config.vocab_word_size = vocab_word_size
        config.num_steps_story_char = story_char_maxlen
        config.num_steps_story_word = story_word_maxlen

        with tf.Graph().as_default() as g:
            model = NeuralModel(config)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

            with tf.Session() as session:
                session.run(init)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                model_dir = os.path.join("logs_Char2Word/", task, str(timestamp))
                writer = tf.summary.FileWriter(model_dir, graph=g)
                # saver.restore(session, "logs_Char2Word/qa1_single-supporting-fact/good/model")
                for epoch in range(config.max_epochs):
                    if verbose:
                        print('Epoch {}'.format(epoch))

                    train_loss, val_loss, train_acc, val_acc, prediction, mask, correct, labels = model.run_epoch(
                        session, (X, Y, X_length, Y_length, Indices_word, Indices_sentence),
                        train_op=model.train_step)

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
                        print([idx_word[x] for x in prediction[0][31]])
                        print([idx_word[x] for x in labels[0][31]])
                        print(mask[0][31])
                        print(correct[0][31])
                    if epoch % 20 == 0:
                        save_path = saver.save(session, os.path.join(model_dir, "model"))
                        print("Model saved in file: %s" % save_path)

                test_acc = model.predict(session, (tX, tY, tX_length, tY_length, tIndices_word, tIndices_sentence))
                print('Testing acc: {}'.format(test_acc))
