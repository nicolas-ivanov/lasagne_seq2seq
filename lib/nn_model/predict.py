import numpy as np

from configs.config import TOKEN_REPRESENTATION_SIZE, TRAIN_BATCH_SIZE, ANSWER_MAX_TOKEN_LENGTH, TEMPERATURE_VALUES, \
    INPUT_SEQUENCE_LENGTH, VOCAB_MAX_SIZE, DEBUG_OUTPUT
from lib.dialog_processor import EOS_SYMBOL, EMPTY_TOKEN, START_TOKEN, get_input_sequence
from lib.w2v_model.vectorizer import get_token_vector
from utils.utils import get_logger

_logger = get_logger(__name__)


def _sample(probs, temperature=1.0):
    """
    helper function to sample an index from a probability array
    """
    strethced_probs = np.log(probs) / temperature
    strethced_probs = np.exp(strethced_probs) / np.sum(np.exp(strethced_probs))
    idx = np.random.choice(strethced_probs.shape[0], p=strethced_probs)
    idx_prob = strethced_probs[idx]
    return idx, idx_prob


def _is_good_token_sequence(token_sequence):
    return EMPTY_TOKEN not in token_sequence and token_sequence[-1] == EOS_SYMBOL


def _predict_sequence(x_batch, nn_model, index_to_token, temperature):
    response = []
    tokens_probs = []

    curr_y_batch = np.zeros((1, ANSWER_MAX_TOKEN_LENGTH), dtype=np.int32)
    if len(x_batch.shape) == 1:
        x_batch = x_batch[np.newaxis, :]

    next_token = START_TOKEN
    i = 0

    while next_token != EOS_SYMBOL and len(response) < ANSWER_MAX_TOKEN_LENGTH - 1:
        probs_batch = nn_model.predict(x_batch, curr_y_batch)

        # probs_batch has shape (batch_size * seq_len, vocab_size)
        # but here batch_size == 1
        # we only need the i-th prediction, so take it
        curr_token_prob_dist = probs_batch[i]
        next_token_id, next_token_prob = _sample(curr_token_prob_dist, temperature)

        next_token = index_to_token[next_token_id]
        response.append(next_token)
        tokens_probs.append(next_token_prob)

        i += 1
        curr_y_batch[0][i] = next_token_id

        if DEBUG_OUTPUT:
            # print all sequences for debugging
            print 'cur sample from prob_batch on token %d: ' % i,
            for d in probs_batch:
                next_token_id, next_token_prob = _sample(d, temperature)
                next_token = index_to_token[next_token_id]
                print next_token,

            print

    response_perplexity = get_sequence_perplexity(tokens_probs)

    return response, response_perplexity


def get_nn_response(x_batch, nn_model, index_to_token, temperature=0.5):
    tokens_sequence, perplexity_sequence = _predict_sequence(x_batch, nn_model, index_to_token, temperature)
    predicted_sentence = ' '.join(tokens_sequence)

    return predicted_sentence, perplexity_sequence


def get_responses_for_temperatures(x_batch, nn_model, index_to_token):
    """
    :return: predictions with different temperature factor for the same sentence
    """
    res_sentences = []
    perplexities = []

    for temperature in TEMPERATURE_VALUES:
        res_sent, curr_perplexity = get_nn_response(x_batch, nn_model, index_to_token, temperature)
        res_sentences.append(res_sent)
        perplexities.append(curr_perplexity)

    return res_sentences, perplexities


def get_sequence_perplexity(predicted_tokens_probabilities):
    """
    note: keep this function in predict.py module to avoid cyclic modules dependence
    """
    sequence_length = len(predicted_tokens_probabilities)
    sequence_perplexity = np.power(2, -np.log2(predicted_tokens_probabilities).sum() / sequence_length)

    return sequence_perplexity

