import cPickle as pickle
import os

import lasagne
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, LSTMLayer, EmbeddingLayer, ReshapeLayer, \
    get_output, get_all_params, get_all_param_values, set_all_param_values
from lasagne.objectives import categorical_crossentropy

from configs.config import HIDDEN_LAYER_DIMENSION, TOKEN_REPRESENTATION_SIZE, GRAD_CLIP, NN_MODEL_PATH
from utils.utils import get_logger

_logger = get_logger(__name__)


class Lasagne_Seq2seq:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.net = self._get_net()
        self.train = self._get_train_fun()
        self.encode = self._get_encoder_fun()
        self.decode = self._get_decoder_fun()
    
    def _get_net(self):
        net = dict()

        net['l_in_x'] = InputLayer(shape=(None, None),
                            input_var=T.imatrix(name="enc_ix"),
                            name="encoder_seq_ix")

        net['l_in_y'] = InputLayer(shape=(None, None),
                            input_var=T.imatrix(name="dec_ix"),
                            name="decoder_seq_ix")

        net['l_emb_x'] = EmbeddingLayer(
            incoming=net['l_in_x'],
            input_size=self.vocab_size,
            output_size=TOKEN_REPRESENTATION_SIZE,
            name="x_embedding")

        net['l_emb_y'] = EmbeddingLayer(
            incoming=net['l_in_y'],
            input_size=self.vocab_size,
            output_size=TOKEN_REPRESENTATION_SIZE,
            name="y_embedding",
            W=net['l_emb_x'].W)

        # encoder ###############################################
        net['l_enc'] = LSTMLayer(
            incoming=net['l_emb_x'],
            num_units=HIDDEN_LAYER_DIMENSION,
            grad_clipping=GRAD_CLIP,
            only_return_final=True)

        # decoder ###############################################
        net['l_dec'] = LSTMLayer(
            incoming=net['l_emb_y'],
            num_units=HIDDEN_LAYER_DIMENSION,
            hid_init=net['l_enc'],
            grad_clipping=GRAD_CLIP,
            name="decoder_lstm")

        # output ###############################################
        net['l_dec_long'] = ReshapeLayer(net['l_dec'], shape=(-1, HIDDEN_LAYER_DIMENSION))

        net['l_dist'] = DenseLayer(
            incoming=net['l_dec_long'],
            num_units=self.vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="dense_output_probas")
        
        return net


    def _get_train_fun(self):
        output_probs = get_output(self.net['l_dist'])

        cost = categorical_crossentropy(
            predictions=output_probs,
            targets=self.net['l_in_y'].input_var
        ).mean()

        all_params = get_all_params(self.net['l_dist'], trainable=True)

        print("Computing train updates...")
        updates = lasagne.updates.rmsprop(
            loss_or_grads=cost,
            params=all_params,
            learning_rate=0.1
        )

        print("Compiling train function...")
        train_fun = theano.function(
            inputs=[self.net['l_in_x'].input_var, self.net['l_in_y'].input_var],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True
        )

        return train_fun


    def _get_encoder_fun(self):
        enc_input = self.net['l_in_x'].input_var
        though_vector = get_output(self.net['l_enc'])

        encoder_fun = theano.function(
            inputs=[enc_input],
            outputs=though_vector
        )

        return encoder_fun


    def _get_decoder_fun(self):
        def get_decoder_1step_net(prev_state, inp_token):
            """
            build nn that represents 1 step of training decoder
            """
            emb_token = EmbeddingLayer(
                incoming=inp_token,
                input_size=self.vocab_size,
                output_size=TOKEN_REPRESENTATION_SIZE,
                W=self.net['l_emb_y'].W,
                name="y_embedding")

            l_dec = LSTMLayer(
                incoming=emb_token,
                num_units=HIDDEN_LAYER_DIMENSION,
                hid_init=prev_state,
                grad_clipping=GRAD_CLIP,
                nonlinearity=lasagne.nonlinearities.tanh,
                only_return_final=True,
                name="decoder_lstm")

            l_dec_long = ReshapeLayer(l_dec, shape=(-1, HIDDEN_LAYER_DIMENSION))

            l_dist = DenseLayer(
                incoming=l_dec_long,
                num_units=self.vocab_size,
                nonlinearity=lasagne.nonlinearities.softmax,
                name="dense_output_probas")

            return l_dist, l_dec

        def set_decoder_weights(dec_1step_probas, dec_1step_next_state):
            """
            set 1step weights equal to training decoder/probas_predictor weights
            """
            params_1step = get_all_params((dec_1step_probas, dec_1step_next_state))
            params_full = get_all_params(self.net['l_dist'])
            params_full_dict = {p.name: p for p in params_full}

            for param_1step in params_1step:
                param_1step.set_value(params_full_dict[param_1step.name].get_value())
                

        prev_state = InputLayer((None, HIDDEN_LAYER_DIMENSION), name="prev_decoder_state")
        inp_token = InputLayer((None, 1), name="prev_decoder_idx", input_var=T.imatrix(name="enc_1step_ix"))

        dec_1step_probas, dec_1step_next_state = get_decoder_1step_net(prev_state, inp_token)
        set_decoder_weights(dec_1step_probas, dec_1step_next_state)

        # compile
        dec_1step_fun = theano.function(
            inputs=[inp_token.input_var, prev_state.input_var],
            outputs=get_output([dec_1step_probas, dec_1step_next_state])
        )

        return dec_1step_fun

    def load_weights(self, model_path):
        with open(model_path, 'r') as f:
            data = pickle.load(f)
        set_all_param_values(model_path, data)

    def save_weights(self, save_path):
        data = get_all_param_values(self)
        with open(save_path, 'w') as f:
            pickle.dump(data, f)


def get_nn_model(vocab_size):
    _logger.info('Initializing NN model with the following params:')
    _logger.info('NN input dimension: %s (token vector size)' % TOKEN_REPRESENTATION_SIZE)
    _logger.info('NN hidden dimension: %s' % HIDDEN_LAYER_DIMENSION)
    _logger.info('NN output dimension: %s (dict size)' % vocab_size)

    model = Lasagne_Seq2seq(vocab_size)

    if os.path.isfile(NN_MODEL_PATH):
        _logger.info('Loading previously calculated weights...')
        model.load_weights(NN_MODEL_PATH)
    else:
        _logger.info("Can't find previously calculated model, so will train it from scratch")

    _logger.info('Model is built\n')

    return model




