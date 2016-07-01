import codecs
import matplotlib
import time
from collections import namedtuple

from lib.dialog_processor import get_input_sequence

matplotlib.use('Agg')

import numpy as np
import unicodecsv as csv
from matplotlib import pyplot

from lib.nn_model.predict import get_nn_response, get_responses_for_temperatures
from lib.w2v_model.vectorizer import get_token_vector
from lib.dialog_processor import EMPTY_TOKEN, PAD_TOKEN
from utils.utils import get_formatted_time, get_git_revision_short_hash, get_logger
from configs.config import RUN_DATE, TEST_DATASET_PATH, DEFAULT_TEMPERATURE, \
    TEST_RESULTS_PATH, BIG_TEST_RESULTS_PATH, PERPLEXITY_LOG_PATH, PERPLEXITY_PIC_PATH, NN_MODEL_PARAMS_STR, \
    TEMPERATURE_VALUES, SMALL_TEST_DATASET_SIZE, LOSS_LOG_PATH, LOSS_PIC_PATH, INPUT_SEQUENCE_LENGTH, \
    TOKEN_REPRESENTATION_SIZE

_logger = get_logger(__name__)

StatsInfo = namedtuple('StatsInfo', 'start_time, iteration_num, sents_batches_num, perplexity')


def get_test_dataset():
    with codecs.open(TEST_DATASET_PATH, 'r', 'utf-8') as fh:
        test_dataset = [s.strip() for s in fh.readlines()]

    return test_dataset


def transform_lines_to_ids(lines_to_transform, token_to_index, max_sent_len, reversed=False):
    """
    :param lines_to_transform: list of lists of tokens to transform to ids
    :param token_to_index: dict that maps each token to its id
    :param max_sent_len:
    :return: X -- numpy array, dtype=np.int32, shape = (len(lines_to_transform), max_sent_len).
    I-th row contains transformed first max_set_len tokens of i-th line of lines_to_transform.
    The rest of each line is ignored.
    if length of a line is less that max_sent_len, it's padded with token_to_index[PAD_TOKEN]
    """
    lines_to_transform = list(lines_to_transform) # transform generator into list if necessary
    n_dialogs = len(lines_to_transform)
    X = np.ones((n_dialogs, max_sent_len), dtype=np.int32) * token_to_index[PAD_TOKEN]

    for i, line in enumerate(lines_to_transform[:-1]):
        for j, token in enumerate(line):
            if j >= max_sent_len:
                break

            id_to_use = token_to_index[token] if token in token_to_index else token_to_index[EMPTY_TOKEN]
            if reversed:
                X[i, max_sent_len - 1 - j] = id_to_use
            else:
                X[i, j] = id_to_use

    return X


def _get_iteration_stats(stats_info):
    stats_str = 'Sents batch iteration number %s\n' % str(stats_info.iteration_num)

    total_elapsed_time = time.time() - stats_info.start_time
    stats_str += 'Total elapsed time: %s\n' % get_formatted_time(total_elapsed_time)

    elapsed_time_per_iteration = total_elapsed_time / stats_info.iteration_num
    stats_str += 'Elapsed time for sents batch iteration: %s\n' % get_formatted_time(elapsed_time_per_iteration)

    estimated_time_for_full_pass = elapsed_time_per_iteration * stats_info.sents_batches_num
    stats_str += 'Estimated time for one full dataset pass: %s\n' % get_formatted_time(estimated_time_for_full_pass)

    stats_str += 'Current perplexity: %s' % stats_info.perplexity

    return stats_str


def init_csv_writer(fh):
    csv_writer = csv.writer(fh, encoding='UTF-8')
    csv_writer.writerow([NN_MODEL_PARAMS_STR])
    csv_writer.writerow(['default diversity: ', DEFAULT_TEMPERATURE])
    csv_writer.writerow(['commit hash: %s' % get_git_revision_short_hash().strip()])
    csv_writer.writerow(['date: %s' % RUN_DATE])

    return csv_writer


def _log_predictions_with_temperatures(sentences, sentences_ids, nn_model, index_to_token, stats_info=None):
    with open(TEST_RESULTS_PATH, 'wb') as test_res_fh:
        csv_writer = init_csv_writer(test_res_fh)

        if stats_info:
            csv_writer.writerow([_get_iteration_stats(stats_info)])

        head_row = ['', 'input sentence'] + ['diversity ' + str(v) for v in TEMPERATURE_VALUES]
        csv_writer.writerow(head_row)

        for sent, sent_ids in zip(sentences, sentences_ids):
            predictions, perplexities = get_responses_for_temperatures(sent_ids, nn_model, index_to_token)
            csv_writer.writerow(['', sent] + predictions + perplexities)


def compute_avr_perplexity(nn_model, validation_ids, index_to_token):
    perplexities = []

    for x_batch in validation_ids:
        _, response_perplexity = get_nn_response(x_batch, nn_model, index_to_token, DEFAULT_TEMPERATURE)
        perplexities.append(response_perplexity)

    return np.mean(perplexities)

def log_perplexity(zipped_vals):
    """
    zipped_vals contains:
    - validation_time_values
    - validation_perplexity_values
    - training_time_values
    - training_perplexity_values
    """
    with codecs.open(PERPLEXITY_LOG_PATH + '.csv', 'w', 'utf-8') as perplexity_fh:
        vals = [[vt, vp, tr, tv] for (vt, vp, tr, tv) in zipped_vals]
        csv_writer = csv.writer(perplexity_fh)
        csv_writer.writerows(vals)


def plot_perplexities(perplexity_stamps):
    fig = pyplot.figure()
    fig.suptitle(NN_MODEL_PARAMS_STR)

    # create subplot grid with 1 row, 1 column and add the current subplot to the first position
    ax = fig.add_subplot(111)
    ax.set_xlabel('hours elapsed')
    ax.set_ylabel('perplexity')
    recent_stamps_num = 100
    recent_perplexity_num = 10

    validation_time_values = [stamp[0] for stamp in perplexity_stamps['validation']]
    validation_perplexity_values = [stamp[1] for stamp in perplexity_stamps['validation']]

    training_time_values = [stamp[0] for stamp in perplexity_stamps['training']]
    training_perplexity_values = [stamp[1] for stamp in perplexity_stamps['training']]

    ax.grid(True)
    ax.plot(validation_time_values[-recent_stamps_num:], validation_perplexity_values[-recent_stamps_num:],
            'b-o',label='perplexity on validation set')
    ax.plot(training_time_values[-recent_stamps_num:], training_perplexity_values[-recent_stamps_num:],
            'r-o', label='perplexity on training set')
    ax.legend()

    recent_perplexity = validation_perplexity_values[-recent_perplexity_num:]
    y_mean = np.mean(recent_perplexity)
    ax.axhline(y_mean)
    fig.savefig(PERPLEXITY_PIC_PATH)

    # also save this data in text format
    zipped_vals = zip(validation_time_values, validation_perplexity_values, training_time_values, training_perplexity_values)
    log_perplexity(zipped_vals)


def update_perplexity_stamps(perplexity_stamps, nn_model, x_val, index_to_token, start_time):
    perplexity = compute_avr_perplexity(nn_model, x_val, index_to_token)
    elapsed_hours = (time.time() - start_time) / (60 * 60)
    perplexity_stamps.append((elapsed_hours, perplexity))


def log_predictions(sentences, sentences_ids, nn_model, index_to_token, stats_info=None):
    with open(BIG_TEST_RESULTS_PATH, 'wb') as test_res_fh:
        csv_writer = init_csv_writer(test_res_fh)

        if stats_info:
            csv_writer.writerow([_get_iteration_stats(stats_info)])

        for sent, sent_ids in zip(sentences, sentences_ids):
            prediction, perplexity = get_nn_response(sent_ids, nn_model, index_to_token)

            csv_writer.writerow(['', sent, prediction, str(perplexity)])


def save_test_results(nn_model, index_to_token, token_to_index, start_time, current_batch_idx, all_batches_num,
                      perplexity_stamps):
    _logger.info('Saving current test results...')
    plot_perplexities(perplexity_stamps)
    cur_perplexity_val = perplexity_stamps['validation'][-1][1]
    cur_perplexity_train = perplexity_stamps['training'][-1][1]
    _logger.info('Current perplexity: train = %0.4f, validation = %0.4f' %
                 (cur_perplexity_train, cur_perplexity_val))

    stats_info = StatsInfo(start_time, current_batch_idx, all_batches_num, cur_perplexity_val)

    test_dataset = get_test_dataset()
    test_dataset_ids = transform_lines_to_ids(test_dataset, token_to_index, INPUT_SEQUENCE_LENGTH, reversed=True)

    log_predictions(test_dataset, test_dataset_ids, nn_model, index_to_token, stats_info)

    small_test_dataset = test_dataset[:SMALL_TEST_DATASET_SIZE]
    small_test_dataset_ids = transform_lines_to_ids(test_dataset, token_to_index, INPUT_SEQUENCE_LENGTH, reversed=True)
    _log_predictions_with_temperatures(small_test_dataset, small_test_dataset_ids, nn_model, index_to_token, stats_info)




def plot_loss(loss_stamps):
    fig = pyplot.figure()
    fig.suptitle(NN_MODEL_PARAMS_STR)

    # create subplot grid with 1 row, 1 column and add the current subplot to the first position
    ax = fig.add_subplot(111)
    ax.set_xlabel('hours elapsed')
    ax.set_ylabel('perplexity')
    recent_stamps_num = 100
    recent_avr_num = 10

    time_values = [stamp[0] for stamp in loss_stamps]
    loss_values = [stamp[1] for stamp in loss_stamps]

    ax.grid(True)
    ax.plot(time_values[-recent_stamps_num:], loss_values[-recent_stamps_num:], 'b-o',label='training loss')
    ax.legend()

    recent_perplexity = loss_values[-recent_avr_num:]
    y_mean = np.mean(recent_perplexity)
    ax.axhline(y_mean)
    fig.savefig(LOSS_PIC_PATH)

    # also save this data in text format
    with codecs.open(LOSS_LOG_PATH + '.csv', 'w', 'utf-8') as loss_fh:
        vals = [[t, l] for (t, l) in zip(time_values, loss_values)]
        csv_writer = csv.writer(loss_fh)
        csv_writer.writerows(vals)


def transform_w2v_model_to_matrix(w2v_model, index_to_token):
    all_words = index_to_token.values()
    all_ids = index_to_token.keys()
    token_to_index = dict(zip(all_words, all_ids))
    n_words = len(index_to_token)
    output = np.zeros((n_words, TOKEN_REPRESENTATION_SIZE))
    for word in all_words:
        idx = token_to_index[word]
        output[idx] = get_token_vector(word, w2v_model)
    return output

