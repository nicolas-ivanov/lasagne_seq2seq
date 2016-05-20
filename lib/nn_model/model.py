import lasagne
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, LSTMLayer, EmbeddingLayer, ReshapeLayer, get_output, get_all_params
from lasagne.objectives import categorical_crossentropy

from configs.config import HIDDEN_LAYER_DIMENSION, TOKEN_REPRESENTATION_SIZE, GRAD_CLIP
from utils.utils import get_logger

_logger = get_logger(__name__)


class Model:
    def __init__(self, vocab_size):
        self.net = dict()

        self.net['l_in_x'] = InputLayer(shape=(None, None),
                            input_var=T.imatrix(name="enc_ix"),
                            name="encoder_seq_ix")

        self.net['l_in_y'] = InputLayer(shape=(None, None),
                            input_var=T.imatrix(name="dec_ix"),
                            name="decoder_seq_ix")

        self.net['l_emb_x'] = EmbeddingLayer(
            incoming=self.net['l_in_x'],
            input_size=vocab_size,
            output_size=TOKEN_REPRESENTATION_SIZE,
            name="x_embedding")

        self.net['l_emb_y'] = EmbeddingLayer(
            incoming=self.net['l_in_y'],
            input_size=vocab_size,
            output_size=TOKEN_REPRESENTATION_SIZE,
            name="y_embedding",
            W=self.net['l_emb_x'].W)

        # encoder ###############################################
        self.net['l_enc'] = LSTMLayer(
            incoming=self.net['l_emb_x'],
            num_units=HIDDEN_LAYER_DIMENSION,
            grad_clipping=GRAD_CLIP,
            only_return_final=True)

        # decoder ###############################################
        self.net['l_dec'] = LSTMLayer(
            incoming=self.net['l_emb_y'],
            num_units=HIDDEN_LAYER_DIMENSION,
            hid_init=self.net['l_enc'],
            grad_clipping=GRAD_CLIP,
            name="decoder_lstm")

        # output ###############################################
        self.net['l_dec_long'] = ReshapeLayer(self.net['l_dec'], shape=(-1, HIDDEN_LAYER_DIMENSION))

        self.net['l_dist'] = DenseLayer(
            incoming=self.net['l_dec_long'],
            num_units=vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax,
            name="dense_output_probas")


    def get_train_funl(self, vocab_size):
        output_probs = get_output(self.net['l_dist'])

        cost = categorical_crossentropy(
            predictions=output_probs,
            targets=self.net['l_in_y'].input_var
        ).mean()

        all_params = get_all_params(self.net['l_dist'], trainable=True)

        print("Computing updates ...")
        updates = lasagne.updates.rmsprop(
            loss_or_grads=cost,
            params=all_params,
            learning_rate=0.1
        )

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        train = theano.function(
            inputs=[self.net['l_in_x'].input_var, self.net['l_in_y'].input_var],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True)

        return train
