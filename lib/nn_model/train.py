import os
import time
from collections import namedtuple
from itertools import tee

import numpy as np

from configs.config import INPUT_SEQUENCE_LENGTH, ANSWER_MAX_TOKEN_LENGTH, DATA_PATH, SAMPLES_BATCH_SIZE, \
    TEST_PREDICTIONS_FREQUENCY, NN_MODEL_PATH, FULL_LEARN_ITER_NUM, BIG_TEST_PREDICTIONS_FREQUENCY, \
    SMALL_TEST_DATASET_SIZE
from lib.nn_model.model_utils import update_perplexity_stamps, save_test_results, get_test_dataset
from lib.nn_model.predict import get_nn_response
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

        X = np.zeros((len(sents_batch), INPUT_SEQUENCE_LENGTH), dtype=np.int)
        Y = np.zeros((len(sents_batch), ANSWER_MAX_TOKEN_LENGTH), dtype=np.int)

        for s_index, sentence in enumerate(sents_batch):
            if s_index == len(sents_batch) - 1:
                break

            for t_index, token in enumerate(sents_batch[s_index][:INPUT_SEQUENCE_LENGTH]):
                X[s_index, t_index] = token_to_index[token]

            for t_index, token in enumerate(sents_batch[s_index + 1][:ANSWER_MAX_TOKEN_LENGTH]):
                Y[s_index, t_index] = token_to_index[token]

        yield X, Y


def save_model(nn_model):
    model_full_path = os.path.join(DATA_PATH, 'nn_models', NN_MODEL_PATH)
    nn_model.save_weights(model_full_path)


def train_model(nn_model, w2v_model, tokenized_dialog_lines, validation_lines, index_to_token):
    train_lines_subset = validation_lines
    # train_lines_subset = random.sample(tokenized_dialog_lines, len(validation_lines))
    token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))

    test_dataset = get_test_dataset()[:SMALL_TEST_DATASET_SIZE]

    start_time = time.time()
    full_data_pass_num = 0
    batch_id = 1

    tokenized_dialog_lines, saved_iterator = tee(tokenized_dialog_lines)

    dialogs_lines_num = 0
    for _ in tokenized_dialog_lines:
        dialogs_lines_num += 1

    batches_num = dialogs_lines_num / SAMPLES_BATCH_SIZE
    perplexity_stamps = {'validation': [], 'training': []}

    try:
        for full_data_pass_num in xrange(1, FULL_LEARN_ITER_NUM + 1):
            _logger.info('\nFull-data-pass iteration num: ' + str(full_data_pass_num))
            dialog_lines_for_train, saved_iterator = tee(saved_iterator)

            for X_train, Y_train in get_training_batch(w2v_model, dialog_lines_for_train, token_to_index):
                progress = float(batch_id) / batches_num * 100
                print '\nbatch iteration %s / %s (%.2f%%)' % (batch_id, dialogs_lines_num, progress)

                loss = nn_model.train(X_train, Y_train)
                print 'loss %.2f' % loss

                avr_time_per_sample = (time.time() - start_time) / batch_id
                expected_time_per_epoch = avr_time_per_sample * batches_num
                print 'expected time for epoch: %.1f h' % (expected_time_per_epoch / 3600)

                if batch_id % TEST_PREDICTIONS_FREQUENCY == 0:
                    for sent in test_dataset:
                        prediction, perplexity = get_nn_response(sent, nn_model, w2v_model, index_to_token)
                        print '%-50s\t -> \t%s [%s]' % (sent, prediction, perplexity)

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
        save_model(nn_model)
        save_test_results(nn_model, w2v_model, index_to_token, start_time, batch_id, batches_num, perplexity_stamps)
