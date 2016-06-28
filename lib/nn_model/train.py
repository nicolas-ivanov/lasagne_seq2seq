import os
import sys
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

def train_model(nn_model,tokenized_dialog_lines, validation_lines, index_to_token):
    token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))

    test_dataset = get_test_dataset()[:SMALL_TEST_DATASET_SIZE]

    start_time = time.time()
    full_data_pass_num = 1
    batch_id = 1

    # tokenized_dialog_lines is an iterator and only allows sequential access, so get array of train lines
    all_train_lines = list(tokenized_dialog_lines)
    train_lines_num = len(all_train_lines)

    X_ids = transform_lines_to_ids(all_train_lines[0:-1:2], token_to_index, INPUT_SEQUENCE_LENGTH)
    Y_ids = transform_lines_to_ids(all_train_lines[1::2], token_to_index, ANSWER_MAX_TOKEN_LENGTH)
    x_test = transform_lines_to_ids(test_dataset, token_to_index, INPUT_SEQUENCE_LENGTH)
    x_val = transform_lines_to_ids(validation_lines, token_to_index, INPUT_SEQUENCE_LENGTH)

    batches_num = train_lines_num / SAMPLES_BATCH_SIZE
    perplexity_stamps = {'validation': [], 'training': []}
    loss_history = []
    objects_processed = 0

    try:
        for full_data_pass_num in xrange(1, FULL_LEARN_ITER_NUM + 1):
            _logger.info('\nStarting epoche #%d; %d objects processed; time = %0.2f' %
                         (full_data_pass_num, objects_processed, time.time() - start_time))

            for X_train, Y_train in get_training_batch(X_ids, Y_ids):
                loss = nn_model.train(X_train, Y_train)
                objects_processed += X_train.shape[0]
                # for i in xrange(5):
                #     for j in xrange(10):
                #         print index_to_token[X_train[i, j]], ' ',
                #     print '>',
                #     for j in xrange(10):
                #         print index_to_token[Y_train[i, j]], ' ',
                #     print
                #
                # sys.exit(0)

                if batch_id % EVALUATE_AND_DUMP_LOSS_FREQUENCY == 0:
                    loss_history.append((time.time(), loss))

                    progress = float(batch_id) / batches_num * 100
                    avr_time_per_sample = (time.time() - start_time) / batch_id
                    expected_time_per_epoch = avr_time_per_sample * batches_num

                    print '\rbatch iteration: %s / %s (%.2f%%) \t\tloss: %.2f \t\t time per epoch: %.2f h' \
                          % (batch_id, batches_num, progress, loss, expected_time_per_epoch / 3600),

                if batch_id % TEST_PREDICTIONS_FREQUENCY == 1:
                    print '\n', datetime.datetime.now().time()
                    print NN_MODEL_PARAMS_STR, '\n'

                    print 'Test dataset:'
                    for i, sent in enumerate(test_dataset):
                        for t in [0.5, 0.3, 0.1, 0.03, 0.01]:
                            prediction, perplexity = get_nn_response(x_test[i], nn_model, index_to_token, temperature=t)
                            print '%-35s\t --t=%0.3f--> \t[%.2f]\t%s' % (sent, t, perplexity, prediction)
                    print
                    print 'Train dataset:'
                    for i, sent in enumerate(all_train_lines[:SMALL_TEST_DATASET_SIZE:2]):
                        for t in [0.5, 0.3, 0.1, 0.03, 0.01]:
                            prediction, perplexity = get_nn_response(X_ids[i], nn_model, index_to_token, temperature=t)
                            print '%-35s\t --t=%0.3f--> \t[%.2f]\t%s' % (sent, t, perplexity, prediction)


                if batch_id % BIG_TEST_PREDICTIONS_FREQUENCY == 0:
                    plot_loss(loss_history)

                    save_model(nn_model)

                    update_perplexity_stamps(perplexity_stamps['validation'], nn_model, x_val,
                                             index_to_token, start_time)

                    train_ids_subset = random.sample(X_ids, x_val.shape[0])
                    update_perplexity_stamps(perplexity_stamps['training'], nn_model, train_ids_subset,
                                             index_to_token, start_time)

                    cur_perplexity_val = perplexity_stamps['validation'][-1][1]
                    cur_perplexity_train = perplexity_stamps['training'][-1][1]
                    _logger.info('Current perplexity: train = %0.4f, validation = %0.4f' %
                                 (cur_perplexity_train, cur_perplexity_val))

                    save_test_results(nn_model, index_to_token, token_to_index, start_time, batch_id, batches_num,
                                      perplexity_stamps)
                batch_id += 1

    except (KeyboardInterrupt, SystemExit):
        _logger.info('Training cycle is stopped manually')
        _logger.info('Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / full_data_pass_num))
        if len(perplexity_stamps['validation']) > 0:
            save_test_results(nn_model, index_to_token, token_to_index, start_time, batch_id, batches_num,
                              perplexity_stamps)
