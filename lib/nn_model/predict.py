import numpy as np

from configs.config import TOKEN_REPRESENTATION_SIZE, TRAIN_BATCH_SIZE, ANSWER_MAX_TOKEN_LENGTH, TEMPERATURE_VALUES, \
    INPUT_SEQUENCE_LENGTH, VOCAB_MAX_SIZE
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
    idx = np.random.choice(np.arange(VOCAB_MAX_SIZE), p=strethced_probs)
    idx_prob = strethced_probs[idx]
    return idx, idx_prob


def _sequence_to_vector(sentence, w2v_model):
    # Here we need predicted vectors only for one sequence, not for the whole batch,
    # however StatefulRNN works in a such a way that we have to feed predict() function
    # the same number of examples as in our train batch.
    # Then we can use only the first predicted sequence and disregard all the rest.
    # If you have more questions, feel free to address them to https://github.com/farizrahman4u/seq2seq
    X = np.zeros((TRAIN_BATCH_SIZE, ANSWER_MAX_TOKEN_LENGTH, TOKEN_REPRESENTATION_SIZE))

    for t, token in enumerate(sentence):
        X[0, t] = get_token_vector(token, w2v_model)

    return X


def _is_good_token_sequence(token_sequence):
    return EMPTY_TOKEN not in token_sequence and token_sequence[-1] == EOS_SYMBOL


def _predict_sequence(input_sequence, nn_model, w2v_model, index_to_token, temperature):
    token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))
    all_tokens = token_to_index.keys()
    response = []
    tokens_probs = []

    input_ids = []
    for token in input_sequence:
        if token in all_tokens:
            token_id = token_to_index[token]
        else:
            token_id = token_to_index[EMPTY_TOKEN]
        input_ids.append(token_id)

    input_vectors = []
    for token in input_sequence:
        token_vector = get_token_vector(token, w2v_model)
        input_vectors.append(token_vector)

    x_batch = np.zeros((1, INPUT_SEQUENCE_LENGTH, TOKEN_REPRESENTATION_SIZE), dtype=np.float32)

    for i, token_vector in enumerate(input_vectors[-INPUT_SEQUENCE_LENGTH:]):
        x_batch[0][i] = token_vector

    # x_batch = np.fliplr(x_batch)  # reverse inputs

    curr_y_batch = np.zeros((1, ANSWER_MAX_TOKEN_LENGTH, TOKEN_REPRESENTATION_SIZE), dtype=np.float32)
    curr_y_batch[0][0] = get_token_vector(START_TOKEN, w2v_model)

    next_token = START_TOKEN
    i = 0

    while next_token != EOS_SYMBOL and len(response) < ANSWER_MAX_TOKEN_LENGTH-1:
        probs_batch = nn_model.predict(x_batch, curr_y_batch)

        # print probs_batch.shape
        # print probs_batch

        # probs_batch has shape (batch_size * seq_len, vocab_size)
        # we only need the last prediction, so take it
        curr_token_prob_dist = probs_batch[i]
        next_token_id, next_token_prob = _sample(curr_token_prob_dist, temperature)

        # for d in probs_batch:
        #     id, next_token_prob = _sample(d, temperature)
        #     t = index_to_token[id]
        #     print t,
        #
        # print


        next_token = index_to_token[next_token_id]
        response.append(next_token)
        tokens_probs.append(next_token_prob)

        i += 1
        curr_y_batch[0][i] = get_token_vector(next_token, w2v_model)

    response_perplexity = get_sequence_perplexity(tokens_probs)

    return response, response_perplexity


def get_nn_response(sentence, nn_model, w2v_model, index_to_token, temperature=0.5):
    input_sequence = get_input_sequence(sentence)
    tokens_sequence, perplexity_sequence = _predict_sequence(input_sequence, nn_model, w2v_model, index_to_token, temperature)
    predicted_sentence = ' '.join(tokens_sequence)

    return predicted_sentence, perplexity_sequence


def get_responses_for_temperatures(sentence, nn_model, w2v_model, index_to_token):
    """
    :return: predictions with different temperature factor for the same sentence
    """
    res_sentences = []
    perplexities = []

    for temperature in TEMPERATURE_VALUES:
        res_sent, curr_perplexity = get_nn_response(sentence, nn_model, w2v_model, index_to_token, temperature)
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

