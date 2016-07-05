import cPickle as pickle
import os
from collections import OrderedDict

import lasagne
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, LSTMLayer, GRULayer, ReshapeLayer, EmbeddingLayer, \
    get_output, get_all_params, get_all_param_values, set_all_param_values, get_all_layers, get_output_shape, SliceLayer, \
    ConcatLayer, DropoutLayer
from lasagne.objectives import categorical_crossentropy
from lasagne.init import Normal

from configs.config import HIDDEN_LAYER_DIMENSION, TOKEN_REPRESENTATION_SIZE, GRAD_CLIP, NN_MODEL_PATH, LEARNING_RATE, \
    ANSWER_MAX_TOKEN_LENGTH, DROPOUT_RATE ,LEARN_WORD_EMBEDDINGS, USE_GRU, CONSTANTLY_FEED_HIDDEN_STATE
from utils.utils import get_logger

_logger = get_logger(__name__)


class Repeat(lasagne.layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        super(Repeat, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        tensors = [input] * self.n
        stacked = theano.tensor.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim + 1)
        return stacked.dimshuffle(dim)


class Lasagne_Seq2seq:
    def __init__(self, vocab_size, learning_rate=LEARNING_RATE, grad_clip=GRAD_CLIP, init_embedding=Normal()):
        self.vocab_size = vocab_size
        self.lr = learning_rate
        self.gc = grad_clip
        self.W = init_embedding
        if USE_GRU:
            self.rnn_layer = GRULayer
        else:
            self.rnn_layer = LSTMLayer

        if CONSTANTLY_FEED_HIDDEN_STATE:
            self.net = self._get_concat_net() # seq2seq v2
        else:
            self.net = self._get_net() # seq2seq v1
        self.train = self._get_train_fun()
        self.predict = self._get_predict_fun()
        # self.encode = self._get_encoder_fun()
        # self.decode = self._get_decoder_fun()
        # self.embedding = self._get_embedding_fun()
        # self.slicing = self._get_slice_fun()
        # self.decoding = self._get_dec_fun()

    def _get_net(self):
        net = OrderedDict()

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
            W=self.W,
            name="embeddings_layer_x"
        )

        net['l_emb_y'] = EmbeddingLayer(
            incoming=net['l_in_y'],
            input_size=self.vocab_size,
            output_size=TOKEN_REPRESENTATION_SIZE,
            W=self.W,
            name="embeddings_layer_y"
        )
        if not LEARN_WORD_EMBEDDINGS:
            net['l_emb_x'].params[net['l_emb_x'].W].remove('trainable')
            net['l_emb_y'].params[net['l_emb_y'].W].remove('trainable')

        # encoder ###############################################
        net['l_enc'] = self.rnn_layer(
            incoming=net['l_emb_x'],
            num_units=HIDDEN_LAYER_DIMENSION,
            grad_clipping=self.gc,
            only_return_final=True,
            name='lstm_encoder'
        )

        # decoder ###############################################

        net['l_dec'] = self.rnn_layer(
            incoming=net['l_emb_y'],
            num_units=HIDDEN_LAYER_DIMENSION,
            hid_init=net['l_enc'],
            grad_clipping=GRAD_CLIP,
            name='lstm_decoder'
        )

        # decoder returns the batch of sequences of though vectors, each corresponds to a decoded token
        # reshape this 3d tensor to 2d matrix so that the next Dense layer can convert each though vector to
        # probability distribution vector

        # output ###############################################
        # cut off the last prob vectors for every prob sequence:
        # they correspond to the tokens that go after EOS_TOKEN and we are not interested in it

        net['l_slice'] = SliceLayer(
            incoming=net['l_dec'],
            indices=slice(0, -1),  # keep all but the last token
            axis=1,  # sequneces axis
            name='slice_layer'
        )

        net['l_dec_long'] = ReshapeLayer(
            incoming=net['l_slice'],
            shape=(-1, HIDDEN_LAYER_DIMENSION),
            name='reshape_layer'
        )

        net['l_dist'] = DenseLayer(
            incoming=net['l_dec_long'],
            num_units=self.vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="dense_output_probas"
        )

        # don't need to reshape back, can compare this "long" output with true one-hot vectors

        return net

    def _get_dropout_net(self):
        net = OrderedDict()

        net['l_in_x'] = InputLayer(shape=(None, None, TOKEN_REPRESENTATION_SIZE),
                                   input_var=T.tensor3(name="enc_ix"),
                                   name="encoder_seq_ix")

        net['l_in_y'] = InputLayer(shape=(None, None, TOKEN_REPRESENTATION_SIZE),
                                   input_var=T.tensor3(name="dec_ix"),
                                   name="decoder_seq_ix")

        # encoder ###############################################
        net['l_enc'] = LSTMLayer(
            incoming=net['l_in_x'],
            num_units=HIDDEN_LAYER_DIMENSION,
            grad_clipping=GRAD_CLIP,
            only_return_final=True,
            name='lstm_encoder'
        )

        # decoder ###############################################

        net['l_dropout_dec'] = DropoutLayer(
            incoming=net['l_enc'],
            p=DROPOUT_RATE,
            name='l_dropout_dec'
        )

        net['l_dec'] = LSTMLayer(
            incoming=net['l_in_y'],
            num_units=HIDDEN_LAYER_DIMENSION,
            hid_init=net['l_dropout_dec'],
            grad_clipping=GRAD_CLIP,
            name='lstm_decoder'
        )

        # decoder returns the batch of sequences of though vectors, each corresponds to a decoded token
        # reshape this 3d tensor to 2d matrix so that the next Dense layer can convert each though vector to
        # probability distribution vector

        # output ###############################################
        # cut off the last prob vectors for every prob sequence:
        # they correspond to the tokens that go after EOS_TOKEN and we are not interested in it
        net['l_slice'] = SliceLayer(
            incoming=net['l_dec'],
            indices=slice(0, -1),  # keep all but the last token
            axis=1,  # sequneces axis
            name='slice_layer'
        )

        net['l_dec_long'] = ReshapeLayer(
            incoming=net['l_slice'],
            shape=(-1, HIDDEN_LAYER_DIMENSION),  # reshape the layer so that we ca
            name='reshape_layer',
        )

        net['l_dropout_long'] = DropoutLayer(
            incoming=net['l_dec_long'],
            p=DROPOUT_RATE,
            name='l_dropout_long'
        )

        net['l_dist'] = DenseLayer(
            incoming=net['l_dropout_long'],
            num_units=self.vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="dense_output_probas"
        )

        # don't need to reshape back, can compare this "long" output with true one-hot vectors

        return net

    def _get_deep_net(self):
        net = OrderedDict()

        net['l_in_x'] = InputLayer(shape=(None, None, TOKEN_REPRESENTATION_SIZE),
                                   input_var=T.tensor3(name="enc_ix"),
                                   name="encoder_seq_ix")

        net['l_in_y'] = InputLayer(shape=(None, None, TOKEN_REPRESENTATION_SIZE),
                                   input_var=T.tensor3(name="dec_ix"),
                                   name="decoder_seq_ix")

        # encoder ###############################################
        net['l_enc'] = LSTMLayer(
            incoming=net['l_in_x'],
            num_units=HIDDEN_LAYER_DIMENSION,
            grad_clipping=GRAD_CLIP,
            name='lstm_encoder'
        )

        net['l_enc_2'] = LSTMLayer(
            incoming=net['l_enc'],
            num_units=HIDDEN_LAYER_DIMENSION,
            grad_clipping=GRAD_CLIP,
            only_return_final=True,
            name='lstm_encoder_2'
        )

        # decoder ###############################################

        net['l_dec'] = LSTMLayer(
            incoming=net['l_in_y'],
            num_units=HIDDEN_LAYER_DIMENSION,
            hid_init=net['l_enc_2'],
            grad_clipping=GRAD_CLIP,
            name='lstm_decoder'
        )

        net['l_dec_2'] = LSTMLayer(
            incoming=net['l_dec'],
            num_units=HIDDEN_LAYER_DIMENSION,
            grad_clipping=GRAD_CLIP,
            name='lstm_decoder_2'
        )

        # decoder returns the batch of sequences of though vectors, each corresponds to a decoded token
        # reshape this 3d tensor to 2d matrix so that the next Dense layer can convert each though vector to
        # probability distribution vector

        # output ###############################################
        # cut off the last prob vectors for every prob sequence:
        # they correspond to the tokens that go after EOS_TOKEN and we are not interested in it
        net['l_slice'] = SliceLayer(
            incoming=net['l_dec_2'],
            indices=slice(0, -1),  # keep all but the last token
            axis=1,  # sequneces axis
            name='slice_layer'
        )

        net['l_dec_long'] = ReshapeLayer(
            incoming=net['l_slice'],
            shape=(-1, HIDDEN_LAYER_DIMENSION),  # reshape the layer so that we ca
            name='reshape_layer'
        )

        net['l_dist'] = DenseLayer(
            incoming=net['l_dec_long'],
            num_units=self.vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="dense_output_probas"
        )

        # don't need to reshape back, can compare this "long" output with true one-hot vectors

        return net


    def _get_concat_net(self):
        net = OrderedDict()


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
            W=self.W
        )

        net['l_emb_y'] = EmbeddingLayer(
            incoming=net['l_in_y'],
            input_size=self.vocab_size,
            output_size=TOKEN_REPRESENTATION_SIZE,
            W=self.W
        )

        if not LEARN_WORD_EMBEDDINGS:
            net['l_emb_x'].params[net['l_emb_x'].W].remove('trainable')
            net['l_emb_y'].params[net['l_emb_y'].W].remove('trainable')

        # encoder ###############################################
        net['l_enc'] = self.rnn_layer(
            incoming=net['l_emb_x'],
            num_units=HIDDEN_LAYER_DIMENSION,
            grad_clipping=GRAD_CLIP,
            only_return_final=True,
            name='lstm_encoder'
        )

        # decoder ###############################################
        net['l_enc_repeated'] = Repeat(
            incoming=net['l_enc'],
            n=ANSWER_MAX_TOKEN_LENGTH,
            name='repeat_layer'
        )

        net['l_concat'] = ConcatLayer(
            incomings=[net['l_emb_y'], net['l_enc_repeated']],
            axis=2,
            name='concat_layer'
        )

        net['l_dec'] = self.rnn_layer(
            incoming=net['l_concat'],
            num_units=HIDDEN_LAYER_DIMENSION,
            # hid_init=net['l_enc'],
            grad_clipping=GRAD_CLIP,
            name='lstm_decoder'
        )

        # decoder returns the batch of sequences of though vectors, each corresponds to a decoded token
        # reshape this 3d tensor to 2d matrix so that the next Dense layer can convert each though vector to
        # probability distribution vector

        # output ###############################################
        # cut off the last prob vectors for every prob sequence:
        # they correspond to the tokens that go after EOS_TOKEN and we are not interested in it
        net['l_slice'] = SliceLayer(
            incoming=net['l_dec'],
            indices=slice(0, -1),    # keep all but the last token
            axis=1,                     # sequneces axis
            name='slice_layer'
        )

        net['l_dec_long'] = ReshapeLayer(
            incoming=net['l_slice'],
            shape=(-1, HIDDEN_LAYER_DIMENSION),
            name='reshape_layer'
        )

        net['l_dist'] = DenseLayer(
            incoming=net['l_dec_long'],
            num_units=self.vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="dense_output_probas"
        )

        # don't need to reshape back, can compare this "long" output with true one-hot vectors
        
        return net


    def _get_train_fun(self):
        output_probs = get_output(self.net['l_dist'])   # "long" 2d matrix with prob distribution

        # cut off the first ids from every id sequence: they correspond to START_TOKEN, that we are not predicting
        target_ids = self.net['l_in_y'].input_var[:, 1:]
        target_ids_flattened = target_ids.flatten()               # "long" vector with target ids

        cost = categorical_crossentropy(
            predictions=output_probs,
            targets=target_ids_flattened
        ).mean()

        all_params = get_all_params(self.net['l_dist'], trainable=True)

        print("Computing train updates...")
        updates = lasagne.updates.adadelta(
            loss_or_grads=cost,
            params=all_params,
            learning_rate=LEARNING_RATE
        )

        print("Compiling train function...")
        train_fun = theano.function(
            inputs=[self.net['l_in_x'].input_var, self.net['l_in_y'].input_var],
            outputs=cost,
            updates=updates
        )

        return train_fun


    def _get_predict_fun(self):
        output_probs = get_output(self.net['l_dist'], deterministic=True)           # "long" 2d matrix with prob distribution

        print("Compiling predict function...")
        predict_fun = theano.function(
            inputs=[self.net['l_in_x'].input_var, self.net['l_in_y'].input_var],
            outputs=output_probs,
        )

        return predict_fun


    def _get_embedding_fun(self):
        enc_input = self.net['l_in_x'].input_var
        though_vector = get_output(self.net['l_enc'])  # "long" 2d matrix with prob distribution

        encoder_fun = theano.function(
            inputs=[enc_input],
            outputs=though_vector
        )

        return encoder_fun


    def _get_dec_fun(self):
        dec_out = get_output(self.net['l_dec'])  # "long" 2d matrix with prob distribution

        dec_fun = theano.function(
            inputs=[self.net['l_in_x'].input_var, self.net['l_in_y'].input_var],
            outputs=dec_out
        )

        return dec_fun


    def _get_slice_fun(self):
        sliced_tensor = get_output(self.net['l_slice'])  # "long" 2d matrix with prob distribution

        slice_fun = theano.function(
            inputs=[self.net['l_in_x'].input_var, self.net['l_in_y'].input_var],
            outputs=sliced_tensor
        )

        return slice_fun


    def _get_encoder_fun(self):
        enc_input = self.net['l_in_x'].input_var        # 2d matrix with X ids
        though_vector = get_output(self.net['l_enc'])

        encoder_fun = theano.function(
            inputs=[enc_input],
            outputs=though_vector
        )

        return encoder_fun


    def _get_decoder_fun(self):

        def get_decoder_1step_net(prev_state, emb_token):
            """
            build nn that represents 1 step of decoder application
            :param prev_state: matrix of shape (batch_size, HIDDEN_LAYER_DIMENSION), float values
            :param inp_token:  matrix of shape (batch_size, 1), stores id of the previous token
            :return:
                l_dec returns new though_vector, matrix shape (batch_size, HIDDEN_LAYER_DIMENSION)
                l_dist returns prob distribution of the next word, matrix shape (batch_size, vocab_size)
            """
            l_dec = LSTMLayer(
                incoming=emb_token,
                num_units=HIDDEN_LAYER_DIMENSION,
                hid_init=prev_state,
                grad_clipping=GRAD_CLIP,
                nonlinearity=lasagne.nonlinearities.tanh,
                only_return_final=True,
                name="lstm_decoder")

            l_dec_long = ReshapeLayer(l_dec, shape=(-1, HIDDEN_LAYER_DIMENSION))

            l_dist = DenseLayer(
                incoming=l_dec_long,
                num_units=self.vocab_size,
                nonlinearity=lasagne.nonlinearities.softmax,
                name="dense_output_probas")

            return l_dec, l_dist

        def set_decoder_weights(decoder_1step):
            """
            set 1step weights equal to training decoder/probas_predictor weights
            """
            params_1step = get_all_params(decoder_1step)
            params_full = get_all_params(self.net['l_dist'])
            params_full_dict = {p.name: p for p in params_full}

            for param_1step in params_1step:
                # use Theano .get_value() and.set_value() methods, applied to the shared variables
                param_1step.set_value(params_full_dict[param_1step.name].get_value())

            # is it the same as the following ?
            # set_all_param_values(decoder_1step, params_full)


        # putting all together #############################################
        # matrix of previous token ids;
        # in reality it has shape (batch_size, 1) since we are looking only on one previous word
        # why on one and not n?

        prev_state = InputLayer((None, HIDDEN_LAYER_DIMENSION), name="prev_decoder_state")
        inp_token = InputLayer((None, 1, TOKEN_REPRESENTATION_SIZE), name="prev_decoder_idx")


        dec_1step_next_state, dec_1step_probas = get_decoder_1step_net(prev_state, inp_token)

        # lstm, embedding and dense layers in 1-step decoder net have many params
        # that we need to make identical to those from the main net
        # the names of the layers in decoder should be the same as the corresponding layers in the main net
        set_decoder_weights(decoder_1step=dec_1step_probas)

        state, probas = get_output([dec_1step_next_state, dec_1step_probas])

        # compile
        dec_1step_fun = theano.function(
            inputs=[prev_state.input_var, inp_token.input_var],
            outputs=[state, probas]
        )

        return dec_1step_fun


    def load_weights(self, model_path):
        with open(model_path, 'r') as f:
            data = pickle.load(f)
        set_all_param_values(self.net['l_dist'], data)

    def save_weights(self, save_path):
        data = get_all_param_values(self.net['l_dist'])
        with open(save_path, 'w') as f:
            pickle.dump(data, f)

    def print_layer_shapes(self):
        print '\n', '-'*100
        print 'Net shapes:\n'

        layers = get_all_layers(self.net['l_dist'])
        for l in layers:
            print '%-20s \t%s' % (l.name, get_output_shape(l))
        print '\n', '-'*100


def get_nn_model(vocab_size, w2v_matrix=None):
    _logger.info('Initializing NN model with the following params:')
    _logger.info('NN input dimension: %s (token vector size)' % TOKEN_REPRESENTATION_SIZE)
    _logger.info('NN hidden dimension: %s' % HIDDEN_LAYER_DIMENSION)
    _logger.info('NN output dimension: %s (dict size)' % vocab_size)

    if not w2v_matrix is None:
        model = Lasagne_Seq2seq(vocab_size, w2v_matrix.astype(theano.config.floatX))
    else:
        model = Lasagne_Seq2seq(vocab_size)

    if os.path.isfile(NN_MODEL_PATH):
        _logger.info('Loading previously calculated weights...')
        model.load_weights(NN_MODEL_PATH)
    else:
        _logger.info("Can't find previously calculated model, so will train it from scratch")

    _logger.info('Model is built\n')
    model.print_layer_shapes()
    return model




