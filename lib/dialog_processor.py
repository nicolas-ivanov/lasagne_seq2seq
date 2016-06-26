import codecs
import json
import os
from collections import Counter
from itertools import tee

import nltk.tokenize

from configs.config import VOCAB_MAX_SIZE
from utils.utils import IterableSentences, get_logger

_PUNKT_MARKS = {'.', '?', '!'}
_tokenizer = nltk.tokenize.RegexpTokenizer(pattern=r'[#]+|[\w\$]+|[^\w\s]')

EOS_SYMBOL = '$$$'
EMPTY_TOKEN = '###'
START_TOKEN = '&&&'
PAD_TOKEN = '___'

_logger = get_logger(__name__)


def get_tokens_voc(tokenized_dialog_lines):
    """
    :param tokenized_dialog_lines: iterator for the efficient use of RAM
    """
    token_counter = Counter()

    for line in tokenized_dialog_lines:
        for token in line:
            token_counter.update([token])

    freq_tokens = [token for token, _ in token_counter.most_common()[:VOCAB_MAX_SIZE-2]]
    token_voc = [PAD_TOKEN, EMPTY_TOKEN] + freq_tokens   # PAD_TOKEN should have id = 0

    return token_voc


def get_transformed_dialog_lines(tokenized_dialog_lines, tokens_voc):
    for line in tokenized_dialog_lines:
        transformed_line = []

        for token in line:
            if token not in tokens_voc:
                token = EMPTY_TOKEN

            transformed_line.append(token)
        yield transformed_line


def get_tokenized_dialog_lines(iterable_dialog_lines):
    for i, line in enumerate(iterable_dialog_lines):
        tokenized_dialog_line = tokenize(line)
        tokenized_dialog_line = [START_TOKEN] + tokenized_dialog_line + [EOS_SYMBOL]
        if i % 2 == 0:
            yield tokenized_dialog_line


def get_tokenized_dialog_lines_from_processed_corpus(iterable_dialog_lines):
    for line in iterable_dialog_lines:
        tokenized_dialog_line = line.strip().split()
        yield tokenized_dialog_line


def process_corpus(corpus_path):
    iterable_dialog_lines = IterableSentences(corpus_path)

    tokenized_dialog_lines = get_tokenized_dialog_lines(iterable_dialog_lines)
    tokenized_dialog_lines_for_voc, tokenized_dialog_lines_for_transform = tee(tokenized_dialog_lines)

    tokens_voc = get_tokens_voc(tokenized_dialog_lines_for_voc)
    transformed_dialog_lines = get_transformed_dialog_lines(tokenized_dialog_lines_for_transform, tokens_voc)

    _logger.info('Token voc size = ' + str(len(tokens_voc)))
    index_to_token = dict(enumerate(tokens_voc))

    return transformed_dialog_lines, index_to_token


def save_corpus(tokenized_dialog, processed_dialog_path):
    with codecs.open(processed_dialog_path, 'w', 'utf-8') as dialogs_fh:
        for tokenized_sentence in tokenized_dialog:
            sentence = ' '.join(tokenized_sentence)
            dialogs_fh.write(sentence + '\n')


def save_index_to_tokens(index_to_token, token_index_path):
    with codecs.open(token_index_path, 'w', 'utf-8') as token_index_fh:
        json.dump(index_to_token, token_index_fh, ensure_ascii=False)


def get_index_to_token(token_index_path):
    with codecs.open(token_index_path, 'r', 'utf-8') as token_index_fh:
        index_to_token = json.load(token_index_fh)
        index_to_token = {int(k): v for k, v in index_to_token.iteritems()}

    return index_to_token


def get_processed_dialog_lines_and_index_to_token(corpus_path, processed_corpus_path, token_index_path):
    _logger.info('Loading corpus data...')

    if os.path.isfile(processed_corpus_path) and os.path.isfile(token_index_path):
        _logger.info(processed_corpus_path + ' and ' + token_index_path + ' exist, loading files from disk')
        processed_dialog_lines = IterableSentences(processed_corpus_path)
        processed_dialog_lines = get_tokenized_dialog_lines_from_processed_corpus(processed_dialog_lines)
        index_to_token = get_index_to_token(token_index_path)
        return processed_dialog_lines, index_to_token

    # continue here if processed corpus and token index are not stored on the disk
    _logger.info(processed_corpus_path + ' and ' + token_index_path + " don't exist, compute and save it")
    processed_dialog_lines, index_to_token = process_corpus(corpus_path)
    processed_dialog_lines, processed_dialog_lines_for_save = tee(processed_dialog_lines)

    save_index_to_tokens(index_to_token, token_index_path)
    save_corpus(processed_dialog_lines_for_save, processed_corpus_path)

    return processed_dialog_lines, index_to_token


def get_lines_for_validation(validation_set_path, index_to_token):
    with codecs.open(validation_set_path, 'r', 'utf-8') as dataset_fh:
        lines = dataset_fh.readlines()
        lines = [tokenize(line.strip()) for line in lines]
        screened_lines = get_transformed_dialog_lines(lines, index_to_token.values())

    # return true array, not iterator
    lines_for_validation = []
    for line in screened_lines:
        lines_for_validation.append(line)

    return lines_for_validation


def tokenize(text):
    tokens = _tokenizer.tokenize(text.lower())
    return tokens


def get_input_sequence(sentence):
    """
    Prepare chatbot's input by tokenizing the sentence and adding necessary punctuation marks.
    Input: "So what's up, buddy"
    Output: ["so", "what", "'", "s", "up", ",", "buddy", ".", "$$$"]
    """
    if not sentence:
        return [START_TOKEN, EOS_SYMBOL]

    # add a dot to the end of the sent in case there is no punctuation mark
    if sentence[-1] not in _PUNKT_MARKS:
        sentence += '.'

    sequence = [START_TOKEN] + tokenize(sentence) + [EOS_SYMBOL]

    return sequence
