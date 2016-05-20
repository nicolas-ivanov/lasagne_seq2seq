from __future__ import print_function

import lasagne
import numpy as np
import theano
import theano.tensor as T


in_text = np.arange(100)

chars = list(set(in_text))
data_size, vocab_size = len(in_text), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

print(vocab_size)

lasagne.random.set_rng(np.random.RandomState(1))

SEQ_LENGTH = 5
N_HIDDEN = 512
LEARNING_RATE = .01
GRAD_CLIP = 100
PRINT_FREQ = 1000
NUM_EPOCHS = 50
BATCH_SIZE = 3


def gen_data(p, batch_size = BATCH_SIZE, data=in_text):
    x = np.zeros((batch_size, SEQ_LENGTH, vocab_size))
    y = np.zeros((batch_size, SEQ_LENGTH, vocab_size))

    for n in range(batch_size):
        ptr = n
        for i in range(SEQ_LENGTH):
            x[n,i,char_to_ix[data[p+ptr+i]]] = 1.
            y[n,i,char_to_ix[data[p+SEQ_LENGTH+ptr+i]]] = 1
    return x, y


def test_gen_data():
    x, y = gen_data(0, 10, in_text)

    print('>', x[0])
    print('>>', y[0])

    print('>', [ix_to_char[np.argmax(xx)] for xx in x[0]])
    print('>>', [ix_to_char[np.argmax(yy)] for yy in y[0]])


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


def build_network():
    from lasagne.layers import InputLayer, LSTMLayer, ConcatLayer, ReshapeLayer, DenseLayer, get_output, get_all_params
    from lasagne.objectives import categorical_crossentropy
    print("Building network ...")

    # inputs ###############################################
    l_in_x = InputLayer(shape=(BATCH_SIZE, None, vocab_size))
    l_in_y = InputLayer(shape=(BATCH_SIZE, None, vocab_size))

    # encoder ###############################################
    l_enc = LSTMLayer(
        l_in_x, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True)
    
    # decoder ###############################################
    l_repeated_enc = Repeat(l_enc, SEQ_LENGTH)
    l_conc = ConcatLayer([l_in_y, l_repeated_enc], axis=2)

    l_dec = LSTMLayer(
        l_conc, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    # output ###############################################
    l_dec_long = ReshapeLayer(l_dec, shape=(-1, N_HIDDEN))

    l_dist = DenseLayer(
        l_dec_long,
        num_units=vocab_size,
        nonlinearity=lasagne.nonlinearities.softmax)

    l_out = ReshapeLayer(l_dist, shape=(BATCH_SIZE, -1, vocab_size))

    # print(lasagne.layers.get_output_shape(l_out))

    # compilations ###############################################
    target_values = T.btensor3('target_output')
    network_output = get_output(l_out)
    cost = categorical_crossentropy(network_output, target_values).mean()

    all_params = get_all_params(l_out,trainable=True)
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function(
        inputs=[l_in_x.input_var, l_in_y.input_var, target_values],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True)

    compute_cost = theano.function(
        inputs=[l_in_x.input_var, target_values],
        outputs=cost,
        allow_input_downcast=True)

    predict = theano.function(
        inputs=[l_in_x.input_var],
        outputs=network_output,
        allow_input_downcast=True)

    return train, predict, compute_cost



def try_it_out(predict):
    N = 20
    curr_x = 1
    sample_ix = []

    for i in range(N):
        next_x_dist = predict(curr_x)[0]
        curr_x = np.argmax(next_x_dist)
        sample_ix.append(curr_x)

    print(sample_ix)


def main():
    print("Training ...")
    p = 0
    num_epochs = 10

    train, predict, compute_cost = build_network()

    try:
        for it in xrange(data_size * num_epochs / BATCH_SIZE):

            avg_cost = 0
            for _ in range(PRINT_FREQ):
                x,y = gen_data(p)

                #print(p)
                p += SEQ_LENGTH + BATCH_SIZE - 1
                if(p+BATCH_SIZE+SEQ_LENGTH >= data_size):
                    print('Carriage Return')
                    p = 0

                avg_cost += train(x, y)
            print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))

            try_it_out(predict)

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    build_network()
    # main()
