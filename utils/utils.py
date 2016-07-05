from copy import copy
import codecs
import logging
import subprocess


def get_logger(file_name):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(file_name)

    return logger


def get_formatted_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    formatted_time = '%d:%02d:%02d' % (h, m, s)

    return formatted_time


class IterableSentences(object):
    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        for line in codecs.open(self._filename, 'r', 'utf-8'):
            yield line.strip()


def rolling_window(token_sequence, window_size):
    seq_window = token_sequence[:window_size]  # First window
    yield seq_window

    for new_token in token_sequence[window_size:]:  # Subsequent windows
        seq_window[:-1] = seq_window[1:]
        seq_window[-1] = new_token
        yield seq_window


def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


class ModelLoaderException(Exception):
    pass


def tee_nobuffer(iter, n=2):
    result = []
    for i in xrange(n - 1):
        result.append(copy(iter))
    result.append(iter)
    return tuple(result)
