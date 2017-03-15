"""
This module loads the data.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import tensorflow as tf


class Data(object):
    def __init__(self, dataset_path, batch_size):
        self.dataset_dir = os.path.dirname(dataset_path)
        self.batch_size = batch_size

        with open(dataset_path) as f:
            metadata = json.load(f)

        self.dataset_size = metadata['dataset_size']
        self.max_sentence_char_length = metadata['max_sentence_char_length']
        self.max_story_length = metadata['max_story_length']
        self.max_query_char_length = metadata['max_query_char_length']
        self.max_story_char_length = metadata['max_story_char_length']
        self.max_story_word_length = metadata['max_story_word_length']
        self.dataset_size = metadata['dataset_size']
        self.vocab_size_char = metadata['vocab_size_char']
        self.vocab_size_word = metadata['vocab_size_word']
        self.tokens_char = metadata['tokens_char']
        self.tokens_word = metadata['tokens_word']
        self.datasets = metadata['datasets']

    @property
    def steps_per_epoch(self):
        return self.dataset_size // self.batch_size + 1

    def get_input_fn(self, name, num_epochs, shuffle):
        def input_fn():
            features = {
                "story": tf.FixedLenFeature([1, self.max_story_char_length], dtype=tf.int64),
                "query": tf.FixedLenFeature([1, self.max_query_char_length], dtype=tf.int64),
                "answer": tf.FixedLenFeature([], dtype=tf.int64),
            }

            dataset_path = os.path.join(self.dataset_dir, self.datasets[name])
            features = tf.contrib.learn.read_batch_record_features(dataset_path,
                                                                   features=features,
                                                                   batch_size=self.batch_size,
                                                                   randomize_input=shuffle,
                                                                   num_epochs=num_epochs)

            story_char = features['story_char']
            story_word = features['story_word']

            return {'story_char': story_char}, story_word

        return input_fn
