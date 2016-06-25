import os
import random
import time
from collections import namedtuple
from itertools import tee

import datetime
import numpy as np

from configs.config import INPUT_SEQUENCE_LENGTH, ANSWER_MAX_TOKEN_LENGTH, DATA_PATH, SAMPLES_BATCH_SIZE, \
    TEST_PREDICTIONS_FREQUENCY, NN_MODEL_PATH, FULL_LEARN_ITER_NUM, BIG_TEST_PREDICTIONS_FREQUENCY, \
    SMALL_TEST_DATASET_SIZE, NN_MODEL_PARAMS_STR, EVALUATE_AND_DUMP_LOSS_FREQUENCY
from lib.nn_model.model_utils import update_perplexity_stamps, save_test_results, get_test_dataset, plot_loss, \
    transform_lines_to_ids, get_test_dataset_ids
from lib.nn_model.predict import get_nn_response
from lib.w2v_model.vectorizer import get_token_vector
from utils.utils import get_logger

StatsInfo = namedtuple('StatsInfo', 'start_time, iteration_num, sents_batches_num')

_logger = get_logger(__name__)


def log_predictions(sentences, nn_model, index_to_token, stats_info=None):
    for sent in sentences:
        prediction = get_nn_response(sent, nn_model, index_to_token)
        _logger.info('[%s] -> [%s]' % (sent, prediction))


def get_test_senteces(file_path):
    with open(file_path) as test_data_fh:
        test_sentences = test_data_fh.readlines()
        test_sentences = [s.strip() for s in test_sentences]

    return test_sentences


def get_training_batch(X_ids, Y_ids):
    n_objects = X_ids.shape[0]
    n_batches = n_objects / SAMPLES_BATCH_SIZE - 1
    for i in xrange(n_batches):
        start = i * SAMPLES_BATCH_SIZE
        end = (i + 1) * SAMPLES_BATCH_SIZE
        yield X_ids[start:end, :], Y_ids[start:end, :]


def save_model(nn_model):
    print 'Saving model...'
    model_full_path = os.path.join(DATA_PATH, 'nn_models', NN_MODEL_PATH)
    nn_model.save_weights(model_full_path)


def train_model(nn_model, w2v_model, tokenized_dialog_lines, validation_lines, index_to_token):
    token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))

    test_dataset = get_test_dataset()[:SMALL_TEST_DATASET_SIZE]

    start_time = time.time()
    full_data_pass_num = 1
    batch_id = 1

    # tokenized_dialog_lines is an iterator and only allows sequential access, so get array of train lines
    all_train_lines = list(tokenized_dialog_lines)
    train_lines_num = len(all_train_lines)

    X_ids = transform_lines_to_ids(all_train_lines[:-1], token_to_index)
    Y_ids = transform_lines_to_ids(all_train_lines[1:], token_to_index)
    x_test = transform_lines_to_ids(test_dataset, token_to_index)

    batches_num = train_lines_num / SAMPLES_BATCH_SIZE
    perplexity_stamps = {'validation': [], 'training': []}
    loss_history = []

    try:
        for full_data_pass_num in xrange(1, FULL_LEARN_ITER_NUM + 1):
            _logger.info('\nFull-data-pass iteration num: ' + str(full_data_pass_num))

            for X_train, Y_train in get_training_batch(X_ids, Y_ids):

                loss = nn_model.train(X_train, Y_train)

                if batch_id % EVALUATE_AND_DUMP_LOSS_FREQUENCY == 0:
                    loss_history.append((time.time(), loss))

                    progress = float(batch_id) / batches_num * 100
                    avr_time_per_sample = (time.time() - start_time) / batch_id
                    expected_time_per_epoch = avr_time_per_sample * batches_num

                    print '\rbatch iteration: %s / %s (%.2f%%) \t\tloss: %.2f \t\t time per epoch: %.2f h' \
                          % (batch_id, batches_num, progress, loss, expected_time_per_epoch / 3600),

                if batch_id % TEST_PREDICTIONS_FREQUENCY == 0:
                    print '\n', datetime.datetime.now().time()
                    print NN_MODEL_PARAMS_STR, '\n'

                    for i, sent in enumerate(test_dataset):
                        prediction, perplexity = get_nn_response(x_test[i], nn_model, index_to_token)
                        print '%-35s\t -> \t[%.2f]\t%s' % (sent, perplexity, prediction)

                    print '\n'

                if batch_id % BIG_TEST_PREDICTIONS_FREQUENCY == 0:
                    plot_loss(loss_history)

                    save_model(nn_model)

                    update_perplexity_stamps(perplexity_stamps['validation'], nn_model, validation_lines, w2v_model,
                                             index_to_token, start_time)

                    train_lines_subset = random.sample(all_train_lines, len(validation_lines))
                    update_perplexity_stamps(perplexity_stamps['training'], nn_model, train_lines_subset, w2v_model,
                                             index_to_token, start_time)

                    save_test_results(nn_model, index_to_token, start_time, batch_id, batches_num,
                                      perplexity_stamps)
                batch_id += 1

    except (KeyboardInterrupt, SystemExit):
        _logger.info('Training cycle is stopped manually')
        _logger.info('Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / full_data_pass_num))
        save_test_results(nn_model, index_to_token, token_to_index, start_time, batch_id, batches_num, perplexity_stamps)
