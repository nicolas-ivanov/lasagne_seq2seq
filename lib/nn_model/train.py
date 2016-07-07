import os
import random
import time
from collections import namedtuple
from itertools import islice
import datetime

from configs.config import INPUT_SEQUENCE_LENGTH, ANSWER_MAX_TOKEN_LENGTH, DATA_PATH, SAMPLES_BATCH_SIZE, \
    TEST_PREDICTIONS_FREQUENCY, NN_MODEL_PATH, FULL_LEARN_ITER_NUM, BIG_TEST_PREDICTIONS_FREQUENCY, \
    SMALL_TEST_DATASET_SIZE, NN_MODEL_PARAMS_STR, TEMPERATURE_VALUES, ALTERNATE_LINES, REVERSE_INPUT
from lib.nn_model.model_utils import update_perplexity_stamps, save_test_results, plot_loss, \
    transform_lines_to_ids
from lib.nn_model.predict import get_nn_response
from utils.utils import get_logger, tee_nobuffer

StatsInfo = namedtuple('StatsInfo', 'start_time, iteration_num, sents_batches_num')

_logger = get_logger(__name__)


def get_test_senteces(file_path):
    with open(file_path) as test_data_fh:
        test_sentences = test_data_fh.readlines()
        test_sentences = [s.strip() for s in test_sentences]

    return test_sentences


def get_training_batch(X_ids, Y_ids):
    n_objects = X_ids.shape[0]
    n_batches = n_objects / SAMPLES_BATCH_SIZE
    for i in xrange(n_batches):
        start = i * SAMPLES_BATCH_SIZE
        end = (i + 1) * SAMPLES_BATCH_SIZE
        yield X_ids[start:end, :], Y_ids[start:end, :]


def save_model(nn_model):
    _logger.info('Saving model...')
    model_full_path = os.path.join(DATA_PATH, 'nn_models', NN_MODEL_PATH)
    nn_model.save_weights(model_full_path)


def train_model(nn_model,tokenized_dialog_lines, validation_lines, index_to_token):
    token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))

    start_time = time.time()
    epoches_counter = 1
    batch_id = 1

    x_data_iterator, y_data_iterator, iterator_for_validation, iterator_for_len_calc = tee_nobuffer(tokenized_dialog_lines, 4)
    if ALTERNATE_LINES:
        x_data_iterator = islice(x_data_iterator, 0, None, 2)
        y_data_iterator = islice(y_data_iterator, 1, None, 2)
    else:
        x_data_iterator = islice(x_data_iterator, 0, None)
        y_data_iterator = islice(y_data_iterator, 1, None)

    train_dataset_sample = list(islice(iterator_for_validation, 0, SMALL_TEST_DATASET_SIZE * 2, 2))
    train_dataset_sample = [' '.join(x) for x in train_dataset_sample]

    n_dialogs = sum(1 for _ in iterator_for_len_calc)

    _logger.info('Iterating through lines to get input matrix')
    X_ids = transform_lines_to_ids(x_data_iterator, token_to_index, INPUT_SEQUENCE_LENGTH, n_dialogs, reversed=REVERSE_INPUT)
    _logger.info('Iterating through lines to get output matrix')
    Y_ids = transform_lines_to_ids(y_data_iterator, token_to_index, ANSWER_MAX_TOKEN_LENGTH, n_dialogs)
    _logger.info('Iterating through lines to get validation matrix')
    x_val = transform_lines_to_ids(validation_lines, token_to_index, INPUT_SEQUENCE_LENGTH, len(validation_lines),
                                   reversed=REVERSE_INPUT)

    _logger.info('Finished! Start training')

    for i in xrange(5):
        for j in xrange(INPUT_SEQUENCE_LENGTH):
            print index_to_token[x_val[i, j]], ' ',
        print

    print

    perplexity_stamps = {'validation': [], 'training': []}
    loss_history = []
    objects_processed = 0
    total_training_time = 0
    batches_num = X_ids.shape[0] / SAMPLES_BATCH_SIZE

    try:
        for epoches_counter in xrange(1, FULL_LEARN_ITER_NUM + 1):
            _logger.info('\nStarting epoche #%d; %d objects processed; time = %0.2f (training of it = %0.2f)' %
                         (epoches_counter, objects_processed, time.time() - start_time, total_training_time))

            for X_train, Y_train in get_training_batch(X_ids, Y_ids):
                prev_time = time.time()
                loss = nn_model.train(X_train, Y_train)
                total_training_time += time.time() - prev_time
                objects_processed += X_train.shape[0]

                loss_history.append((time.time(), loss))

                progress = float(batch_id) / batches_num * 100
                avr_time_per_sample = (time.time() - start_time) / batch_id
                expected_time_per_epoch = avr_time_per_sample * batches_num

                _logger.info('\rbatch iteration: %s / %s (%.2f%%) \t\tloss: %.2f \t\t time per epoch: %.2f h' \
                      % (batch_id, batches_num, progress, loss, expected_time_per_epoch / 3600))

                if batch_id % TEST_PREDICTIONS_FREQUENCY == 1:
                    _logger.info(str(datetime.datetime.now().time()))
                    _logger.info(NN_MODEL_PARAMS_STR)

                    _logger.info('Test dataset:')
                    for i, sent in enumerate(validation_lines):
                        for t in TEMPERATURE_VALUES:
                            prediction, perplexity = get_nn_response(x_val[i], nn_model, index_to_token, temperature=t)
                            _logger.info('%-35s\t --t=%0.3f--> \t[%.2f]\t%s' % (' '.join(sent), t, perplexity,
                                                                                prediction))

                    _logger.info('Train dataset:')
                    for i, sent in enumerate(train_dataset_sample):
                        for t in TEMPERATURE_VALUES:
                            prediction, perplexity = get_nn_response(X_ids[i], nn_model, index_to_token, temperature=t)
                            _logger.info('%-35s\t --t=%0.3f--> \t[%.2f]\t%s' % (' '.join(sent), t, perplexity,
                                                                                prediction))

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

                    save_test_results(validation_lines, nn_model, index_to_token, token_to_index, start_time, batch_id, batches_num,
                                      perplexity_stamps)
                batch_id += 1

    except (KeyboardInterrupt, SystemExit):
        _logger.info('Training cycle is stopped manually')
        _logger.info('Current time per full-data-pass iteration: %s' % ((time.time() - start_time) / epoches_counter))
        if len(perplexity_stamps['validation']) > 0:
            save_test_results(validation_lines, nn_model, index_to_token, token_to_index, start_time, batch_id, batches_num,
                              perplexity_stamps)
