import numpy as np

from configs.config import TOKEN_REPRESENTATION_SIZE, TRAIN_BATCH_SIZE, ANSWER_MAX_TOKEN_LENGTH
from lib.dialog_processor import EOS_SYMBOL, EMPTY_TOKEN, START_TOKEN
from lib.w2v_model.vectorizer import get_token_vector
from utils.utils import get_logger
from utils.utils import tokenize

_logger = get_logger(__name__)


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


def _predict_sequence(input_sequence, nn_model, index_to_token, temperature):
    token_to_index = dict(zip(index_to_token.values(), index_to_token.keys()))
    answer = []

    input_ids = [token_to_index[token] for token in input_sequence]
    x_batch = [input_ids]
    though_vector = nn_model.encode(x_batch)

    next_token = prev_token = START_TOKEN
    prev_state_batch = though_vector


    i = 0

    while next_token != EOS_SYMBOL and len(answer) < ANSWER_MAX_TOKEN_LENGTH:
        # prev_token_batch = np.array(token_to_index[prev_token], dtype=np.int)[np.newaxis]

        # what the difference?
        prev_token_batch = np.zeros((1,1), dtype=np.int)
        prev_token_batch[0] = token_to_index[prev_token]

        next_state_batch, next_token_probas_batch = \
            nn_model.decode(prev_state_batch, prev_token_batch)

        next_token_id = np.argmax(next_token_probas_batch[0])
        next_token = index_to_token[next_token_id]
        answer.append(next_token)

        prev_token = next_token
        prev_state_batch = next_state_batch

    return answer


def predict_sentence(sentence, nn_model, w2v_model, index_to_token, temperature=0.5):
    input_sequence = tokenize(sentence + ' ' + EOS_SYMBOL)
    tokens_sequence = _predict_sequence(input_sequence, nn_model, index_to_token, temperature)
    predicted_sentence = ' '.join(tokens_sequence)

    return predicted_sentence