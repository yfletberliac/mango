"""
This module loads and pre-processes the bAbI dataset [v1.2] into TFRecords.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import re
import json
import tarfile
import tensorflow as tf

from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('source_dir', 'datasets/', 'Directory containing bAbI sources.')
tf.app.flags.DEFINE_string('target_dir', 'datasets/processed/', 'Where to write datasets.')
tf.app.flags.DEFINE_boolean('include_10k', False, 'Whether to use 10k or 1k examples.')

SPLIT_RE = re.compile('(\W+)?')

PAD_TOKEN = '_PAD'
PAD_ID = 0


def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]


def tokenize_char(sentence):
    """
    Tokenize a string by splitting on characters.
    """
    return list(sentence.lower())


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
            query, answer, supporting = line.split('\t')
            query = tokenize_char(query)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            stories.append((substory, query, answer))
            story.append('')
        else:
            sentence_char = tokenize_char(line)
            sentence_word = tokenize_word(line)
            story_char.append(sentence_char)
            story_word.append(sentence_word)
    return stories


def get_tokenizer_char(stories):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    tokens_all = []
    for story, query, answer in stories:
        tokens_all.extend([token for sentence in story for token in sentence])
        tokens_all.extend([token for sentence in query for token in sentence])
    vocab = [PAD_TOKEN] + sorted(set(tokens_all))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return token_to_id


def get_tokenizer_word(stories):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    tokens_all = []
    for story, query, answer in stories:
        tokens_all.extend([answer])
    vocab = sorted(set(tokens_all))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return token_to_id


def tokenize_stories(stories, token_to_id_char, token_to_id_word):
    """
    Convert all tokens into their unique ids.
    """
    story_ids = []
    for story, query, answer in stories:
        story = [[token_to_id_char[token] for token in sentence] for sentence in story]
        query = [token_to_id_char[token] for token in query]
        answer = token_to_id_word[answer]
        story_ids.append((story, query, answer))
    return story_ids


def get_max_story_char_length(stories):
    char_total_length = []
    for story, _ in stories:
        char_length = []
        for sentence in story:
            char_length.append(len(sentence))
        char_total_length.append(sum(char_length))
    return max(char_total_length)


def get_max_story_word_length(stories):
    word_total_length = []
    for _, story in stories:
        word_length = []
        for sentence in story:
            word_length.append(len(sentence))
        word_total_length.append(sum(word_length))
    return max(word_total_length)


def pad_stories(stories, max_story_char_length, max_story_word_length):
    """
    Pad single vector stories, and queries to a consistence length.
    """
    padded_story = []
    for story_char, story_word in stories:
        story_flat_char = [token_id for sentence in story_char for token_id in sentence]
        story_flat_word = [token_id for sentence in story_word for token_id in sentence]

        for _ in range(max_story_char_length - len(story_flat_char)):
            story_flat_char.append(PAD_ID)
        for _ in range(max_story_word_length - len(story_flat_word)):
            story_flat_word.append(PAD_ID)

        padded_story.append([story_flat_char, story_flat_word])
        assert len(story_flat_char) == max_story_char_length
        assert len(story_flat_word) == max_story_word_length
    return padded_story


def int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_dataset(stories, path):
    """
    Save the stories into TFRecords.
    """
    writer = tf.python_io.TFRecordWriter(path)
    for story_char, story_word in stories:
        features = tf.train.Features(feature={
            'story_char': int64_features(story_char),
            'story_word': int64_features(story_word),
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()


def main():
    if not os.path.exists(FLAGS.target_dir):
        os.makedirs(FLAGS.target_dir)

    filenames = [
        'qa1_single-supporting-fact',
        'qa2_two-supporting-facts',
        'qa3_three-supporting-facts',
        'qa4_two-arg-relations',
        'qa5_three-arg-relations',
        'qa6_yes-no-questions',
        'qa7_counting',
        'qa8_lists-sets',
        'qa9_simple-negation',
        'qa10_indefinite-knowledge',
        'qa11_basic-coreference',
        'qa12_conjunction',
        'qa13_compound-coreference',
        'qa14_time-reasoning',
        'qa15_basic-deduction',
        'qa16_basic-induction',
        'qa17_positional-reasoning',
        'qa18_size-reasoning',
        'qa19_path-finding',
        'qa20_agents-motivations',
    ]

    for filename in tqdm(filenames):
        if FLAGS.include_10k:
            stories_path_train = os.path.join('tasks_1-20_v1-2/en-10k/', filename + '_train.txt')
            stories_path_test = os.path.join('tasks_1-20_v1-2/en-10k/', filename + '_test.txt')
            dataset_path_train = os.path.join(FLAGS.target_dir, filename + '_10k_train.tfrecords')
            dataset_path_test = os.path.join(FLAGS.target_dir, filename + '_10k_test.tfrecords')
            metadata_path = os.path.join(FLAGS.target_dir, filename + '_10k.json')
            dataset_size = 10000
        else:
            stories_path_train = os.path.join('tasks_1-20_v1-2/en/', filename + '_train.txt')
            stories_path_test = os.path.join('tasks_1-20_v1-2/en/', filename + '_test.txt')
            dataset_path_train = os.path.join(FLAGS.target_dir, filename + '_1k_train.tfrecords')
            dataset_path_test = os.path.join(FLAGS.target_dir, filename + '_1k_test.tfrecords')
            metadata_path = os.path.join(FLAGS.target_dir, filename + '_1k.json')
            dataset_size = 1000

        tar = tarfile.open(os.path.join(FLAGS.source_dir, 'babi_tasks_data_1_20_v1.2.tar.gz'))

        f_train = tar.extractfile(stories_path_train)
        f_test = tar.extractfile(stories_path_test)

        stories_train = parse_stories(f_train.readlines())
        stories_test = parse_stories(f_test.readlines())

        token_to_id_char = get_tokenizer_char(stories_train + stories_test)
        token_to_id_word = get_tokenizer_word(stories_train + stories_test)

        stories_token_train = tokenize_stories(stories_train, token_to_id_char, token_to_id_word)
        stories_token_test = tokenize_stories(stories_test, token_to_id_char, token_to_id_word)
        stories_token_all = stories_token_train + stories_token_test

        max_sentence_length = max([len(sentence) for story, _ in stories_token_all for sentence in story])
        max_story_length = max([len(story) for story, _ in stories_token_all])
        max_story_char_length = get_max_story_char_length(stories_token_all)
        max_story_word_length = get_max_story_word_length(stories_token_all, token_to_id_char)
        vocab_size_char = len(token_to_id_char)
        vocab_size_word = len(token_to_id_word)

        with open(metadata_path, 'w') as f:
            metadata = {
                'dataset_name': filename,
                'dataset_size': dataset_size,
                'max_sentence_char_length': max_sentence_length,
                'max_story_length': max_story_length,
                'max_query_char_length': max_query_length,
                'max_story_char_length': max_story_char_length,
                'max_story_word_length': max_story_word_length,
                'vocab_size_char': vocab_size_char,
                'vocab_size_word': vocab_size_word,
                'tokens_char': token_to_id_char,
                'tokens_word': token_to_id_word,
                'datasets': {
                    'train': os.path.basename(dataset_path_train),
                    'test': os.path.basename(dataset_path_test),
                }
            }
            json.dump(metadata, f)

        stories_pad_train = pad_stories(stories_token_train, max_story_char_length, max_story_word_length)
        stories_pad_test = pad_stories(stories_token_test, max_story_char_length, max_story_word_length)

        save_dataset(stories_pad_train, dataset_path_train)
        save_dataset(stories_pad_test, dataset_path_test)


if __name__ == '__main__':
    main()
