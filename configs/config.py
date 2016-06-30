import os
import time


USE_GRU = True  # use GRU cells instead of LSTM cells
CONSTANTLY_FEED_HIDDEN_STATE = True

DATA_PATH = '/var/lib/lasagne_seq2seq'
CORPORA_DIR = 'corpora_raw'
PROCESSED_CORPORA_DIR = 'corpora_processed'
W2V_MODELS_DIR = 'w2v_models'

# set paths of training and testing sets
CORPUS_NAME = 'repeated_phrases'
CORPUS_PATH = os.path.join('data/train', CORPUS_NAME + '.txt')
# CORPUS_NAME = 'dialogs_50mb'
# CORPUS_PATH = os.path.join(DATA_PATH, CORPORA_DIR, CORPUS_NAME + '.txt')
TEST_DATASET_PATH = os.path.join('data', 'test', CORPUS_NAME + '.txt')
SMALL_TEST_DATASET_PATH = os.path.join('data', 'test', CORPUS_NAME + '.txt')

# set word2vec params
TOKEN_REPRESENTATION_SIZE = 200
VOCAB_MAX_SIZE = 10000
INITIALIZE_WORD_EMBEDDINGS_WITH_WORD2VEC = False
USE_PRETRAINED_W2V = False
LEARN_WORD_EMBEDDINGS = True
GLOVE_MODEL_PATH = 'data/glove.6B.200d.txt' # You can find this file here http://nlp.stanford.edu/data/glove.6B.zip

#set seq2seq params
HIDDEN_LAYER_DIMENSION = 512
INPUT_SEQUENCE_LENGTH = 5
ANSWER_MAX_TOKEN_LENGTH = 5

# set training params
TRAIN_BATCH_SIZE = 512
SAMPLES_BATCH_SIZE = TRAIN_BATCH_SIZE
SMALL_TEST_DATASET_SIZE = 5
TEST_PREDICTIONS_FREQUENCY = 500
BIG_TEST_PREDICTIONS_FREQUENCY = 1000
FULL_LEARN_ITER_NUM = 5000

# local paths and strs that depend on previous params
TOKEN_INDEX_PATH = os.path.join(DATA_PATH, 'words_index', 'w_idx_' + CORPUS_NAME + '_v' + str(VOCAB_MAX_SIZE) + '.txt')
PROCESSED_CORPUS_PATH = os.path.join(DATA_PATH, PROCESSED_CORPORA_DIR, CORPUS_NAME + '_v' + str(VOCAB_MAX_SIZE) + '.txt')

# w2v params that depend on previous params
W2V_PARAMS = {
    "use_pretrained": USE_PRETRAINED_W2V,
    "txt_path": GLOVE_MODEL_PATH,
    "corpus_name": CORPUS_NAME,
    "save_path": DATA_PATH,
    "pre_corpora_dir": CORPORA_DIR,
    "new_models_dir": W2V_MODELS_DIR,
    "vect_size": TOKEN_REPRESENTATION_SIZE,
    "vocab_size": VOCAB_MAX_SIZE,
    "win_size": 5,
    "workers_num": 25
}

GRAD_CLIP = 10.0
LEARNING_RATE = 1.0       # hm, what learning rate should be here?
NN_LAYERS_NUM = 1
DROPOUT_RATE = 0.5
DEFAULT_TEMPERATURE = 0.7
TEMPERATURE_VALUES = [0.3, 0.5, 0.8, 1.2]

def get_nn_params_str():
    params_str = '_{cell_type}_{net_type}_ln{layers_num}_hd{hidden_dim}_d{dropout_rate}_cl{cont_len}_lr{learning_rate}_gc_{gradient_clip}'
    params_str = params_str.format(cell_type='gru' if USE_GRU else 'lstm',
                                   net_type='concat' if CONSTANTLY_FEED_HIDDEN_STATE else 'v1',
                                   layers_num=NN_LAYERS_NUM, hidden_dim=HIDDEN_LAYER_DIMENSION,
                                   dropout_rate=DROPOUT_RATE,cont_len=INPUT_SEQUENCE_LENGTH,
                                   learning_rate=LEARNING_RATE, gradient_clip=GRAD_CLIP)

    return params_str

def get_w2v_params_str(params=W2V_PARAMS):
    params_str = '_window{window_size}_voc{voc_size}_vec{vec_size}'
    params_str = params_str.format(window_size=params['win_size'], voc_size=params['vocab_size'], vec_size=params['vect_size'])

    return params_str

def get_model_full_params_str():
    return CORPUS_NAME + get_w2v_params_str() + get_nn_params_str()


NN_MODEL_PARAMS_STR = get_model_full_params_str()

RUN_DATE = time.strftime('_%y.%m.%d_%H:%M_')

TEST_RESULTS_PATH = os.path.join(DATA_PATH, 'results', 'res' + RUN_DATE + NN_MODEL_PARAMS_STR + '.csv')
BIG_TEST_RESULTS_PATH = os.path.join(DATA_PATH, 'results', 'big_res' + RUN_DATE + NN_MODEL_PARAMS_STR + '.csv')
PERPLEXITY_PIC_PATH = os.path.join(DATA_PATH, 'perplexity', 'perplexity' + RUN_DATE + NN_MODEL_PARAMS_STR + '.png')
PERPLEXITY_LOG_PATH = os.path.join(DATA_PATH, 'perplexity', 'perplexity' + RUN_DATE + NN_MODEL_PARAMS_STR + '.csv')
LOSS_PIC_PATH = os.path.join(DATA_PATH, 'perplexity', 'loss' + RUN_DATE + NN_MODEL_PARAMS_STR + '.png')
LOSS_LOG_PATH = os.path.join(DATA_PATH, 'perplexity', 'loss' + RUN_DATE + NN_MODEL_PARAMS_STR + '.csv')
DIALOGS_TEST_RESULTS_PATH = os.path.join(DATA_PATH, 'results', 'dialogs_res_' + RUN_DATE + NN_MODEL_PARAMS_STR + '.csv')

NN_MODEL_PATH = os.path.join(DATA_PATH, 'nn_models', NN_MODEL_PARAMS_STR)