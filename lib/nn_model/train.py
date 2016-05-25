import os
import random
import time
from collections import namedtuple
from itertools import tee

import datetime
import numpy as np

from configs.config import INPUT_SEQUENCE_LENGTH, ANSWER_MAX_TOKEN_LENGTH, DATA_PATH, SAMPLES_BATCH_SIZE, \
    TEST_PREDICTIONS_FREQUENCY, NN_MODEL_PATH, FULL_LEARN_ITER_NUM, BIG_TEST_PREDICTIONS_FREQUENCY, \
    SMALL_TEST_DATASET_SIZE, TOKEN_REPRESENTATION_SIZE
from lib.nn_model.model_utils import update_perplexity_stamps, save_test_results, get_test_dataset
from lib.nn_model.predict import get_nn_response
from lib.w2v_model.vectorizer import get_token_vector
from utils.utils import get_logger

StatsInfo = namedtuple('StatsInfo', 'start_time, iteration_num, sents_batches_num')

_logger = get_logger(__name__)


def log_predictions(sentences, nn_model, w2v_model, index_to_token, stats_info=None):
    for sent in sentences:
        prediction = get_nn_response(sent, nn_model, w2v_model, index_to_token)
        _logger.info('[%s] -> [%s]' % (sent, prediction))


def get_test_senteces(file_path):
    with open(file_path) as test_data_fh:
        test_sentences = test_data_fh.readlines()
        test_sentences = [s.strip() for s in test_sentences]

    return test_sentences


def _batch(tokenized_dialog_lines, batch_size=1):
    batch = []

    for line in tokenized_dialog_lines:
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []

    # return an empty array instead of yielding incomplete batch
    yield []


def get_training_batch(w2v_model, tokenized_dialog, token_to_index):

    for sents_batch in _batch(tokenized_dialog, SAMPLES_BATCH_SIZE):
        if not sents_batch:
            continue

        X = np.zeros((len(sents_batch), INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE), dtype=np.float32)
        Y = np.zeros((len(sents_batch), ANSWER_MAX_TOKEN_LENGTH, TOKEN_REPRESENTATION_SIZE), dtype=np.float32)
        Y_ids = np.zeros((len(sents_batch), ANSWER_MAX_TOKEN_LENGTH), dtype=np.int32)

        for s_index, sentence in enumerate(sents_batch):
            if s_index == len(sents_batch) - 1:
                break

            for t_index, token in enumerate(sents_batch[s_index][:INPUT_SEQUENCE_LENGTH]):
                X[s_index, t_index] = get_token_vector(token, w2v_model)

            for t_index, token in enumerate(sents_batch[s_index + 1][:ANSWER_MAX_TOKEN_LENGTH]):
                Y[s_index, t_index] = get_token_vector(token, w2v_model)
                Y_ids[s_index, t_index] = token_to_index[token]

        X = np.fliplr(X)  # reverse inputs

        yield X, Y, Y_ids


def save_model(nn_model):
    print 'Saving model...'
    model_full_path = os.path.join(DATA_PATH, 'nn_models', NN_MODEL_PATH)
    nn_model.save_weights(model_full_path)


def train_model(nn_model, w2v_model, tokenized_dialog_lines, validation_lines, index_to_token):
    token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))

    test_dataset = get_test_dataset()[:SMALL_TEST_DATASET_SIZE]

    start_time = time.time()
    full_data_pass_num = 0
    batch_id = 1

    tokenized_dialog_lines, saved_iterator = tee(tokenized_dialog_lines)

    all_train_lines = []

    # tokenized_dialog_lines is an iterator and only allows sequential access, so get array of train lines
    for line in tokenized_dialog_lines:
        all_train_lines.append(line)

    train_lines_subset = random.sample(all_train_lines, len(validation_lines))
    train_lines_num = len(all_train_lines)

    batches_num = train_lines_num / SAMPLES_BATCH_SIZE
    perplexity_stamps = {'validation': [], 'training': []}

    try:
        for full_data_pass_num in xrange(1, FULL_LEARN_ITER_NUM + 1):
            _logger.info('\nFull-data-pass iteration num: ' + str(full_data_pass_num))
            lines_for_train, saved_iterator = tee(saved_iterator)

            for X_train, Y_train, Y_ids in get_training_batch(w2v_model, lines_for_train, token_to_index):

                # print X_train[0]
                # print Y_train[0]
                # print Y_ids[0]

                # print nn_model.decoding(X_train, Y_train)
                # print nn_model.slicing(X_train, Y_train)

                loss = nn_model.train(X_train, Y_train, Y_ids)

                progress = float(batch_id) / batches_num * 100
                avr_time_per_sample = (time.time() - start_time) / batch_id
                expected_time_per_epoch = avr_time_per_sample * batches_num

                print '\rbatch iteration: %s / %s (%.2f%%) \t\tloss: %.2f \t\t time per epoch: %.2f h' \
                      % (batch_id, batches_num, progress, loss, expected_time_per_epoch / 3600),

                if batch_id % TEST_PREDICTIONS_FREQUENCY == 0:
                    print '\n%s\n' % datetime.datetime.now().time()

                    for sent in test_dataset:
                        prediction, perplexity = get_nn_response(sent, nn_model, w2v_model, index_to_token)
                        print '%-50s\t -> \t[%.2f]\t%s' % (sent, perplexity, prediction)

                    print '\n'

                if batch_id % BIG_TEST_PREDICTIONS_FREQUENCY == 0:
                    save_model(nn_model)

                    update_perplexity_stamps(perplexity_stamps['validation'], nn_model, validation_lines, w2v_model,
                                             index_to_token, start_time)
                    update_perplexity_stamps(perplexity_stamps['training'], nn_model, train_lines_subset, w2v_model,
                                             index_to_token, start_time)

                    save_test_results(nn_model, w2v_model, index_to_token, start_time, batch_id, batches_num,
                                      perplexity_stamps)
                batch_id += 1

    except (KeyboardInterrupt, SystemExit):
        _logger.info('Training cycle is stopped manually')
        _logger.info('Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / full_data_pass_num))
        save_test_results(nn_model, w2v_model, index_to_token, start_time, batch_id, batches_num, perplexity_stamps)
