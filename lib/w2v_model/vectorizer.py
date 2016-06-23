import numpy as np

from configs.config import TOKEN_REPRESENTATION_SIZE
from lib.dialog_processor import EMPTY_TOKEN, PAD_TOKEN


def get_token_vector(token, model):
    if token is PAD_TOKEN:
        return np.zeros(TOKEN_REPRESENTATION_SIZE, dtype=np.float32)

    if token is PAD_TOKEN:
        return np.zeros(TOKEN_REPRESENTATION_SIZE, dtype=np.float32)

    if token in model.vocab:
        return np.array(model[token])

    if EMPTY_TOKEN in model.vocab:
        return np.array(model[EMPTY_TOKEN])

    return np.zeros(TOKEN_REPRESENTATION_SIZE, dtype=np.float32)


def get_vectorized_token_sequence(sequence, w2v_model, max_sequence_length, reverse=False):
    vectorized_token_sequence = np.zeros((max_sequence_length, TOKEN_REPRESENTATION_SIZE), dtype=np.float)

    for i, word in enumerate(sequence):
        vectorized_token_sequence[i] = get_token_vector(word, w2v_model)

    if reverse:
        # reverse token vectors order in input sequence as suggested here
        # http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
        vectorized_token_sequence = vectorized_token_sequence[::-1]

    return vectorized_token_sequence
