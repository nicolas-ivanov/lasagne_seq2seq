import os
import sys
import argparse
from itertools import tee

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dialog_processor import get_processed_dialog_lines_and_index_to_token, get_lines_for_validation
from configs.config import CORPUS_PATH, PROCESSED_CORPUS_PATH, TOKEN_INDEX_PATH, W2V_PARAMS, SMALL_TEST_DATASET_PATH
from lib.w2v_model import w2v
from lib.nn_model.model import get_nn_model
from lib.nn_model.train import train_model
from lib.nn_model.model_utils import transform_w2v_model_to_matrix
from utils.utils import get_logger

_logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", type=float,
            help="learning rate", default=0.01)
    parser.add_argument("-c", type=float,
            help="gradient clipping", default=100.0)

    args = parser.parse_args()

def learn():
    # preprocess the dialog and get index for its vocabulary
    processed_dialog_lines, index_to_token = \
        get_processed_dialog_lines_and_index_to_token(CORPUS_PATH, PROCESSED_CORPUS_PATH, TOKEN_INDEX_PATH)

    lines_for_validation = get_lines_for_validation(SMALL_TEST_DATASET_PATH, index_to_token)

    # dualize iterator
    dialog_lines_for_w2v, dialog_lines_for_nn = tee(processed_dialog_lines)
    _logger.info('-----')

    # use gensim realisatino of word2vec instead of keras embeddings due to extra flexibility
    w2v_model = w2v.get_dialogs_model(W2V_PARAMS, dialog_lines_for_w2v)
    _logger.info('-----')

    w2v_matrix = transform_w2v_model_to_matrix(w2v_model, index_to_token)
    nn_model = get_nn_model(len(index_to_token), w2v_matrix)
    _logger.info('-----')

    train_model(nn_model, dialog_lines_for_nn, lines_for_validation, index_to_token)


if __name__ == '__main__':
    learn()
