import numpy as np

def get_unique_word_list(filename):
    words_set = set()

    with open(filename) as f:
        for line in f:
            for w in line.split():
                words_set.add(w.strip())
    words_list = []
    for w in words_set:
        w = w.lower()
        if w.isalpha():
            words_list.append(w)
    return words_list


def generate_repeated_phrases(output_filename, words_list, n_lines=10000, n_words_in_line=3):
    with open(output_filename, 'w') as f:
        for i in xrange(n_lines):
            word = np.random.choice(words_list)
            line_to_use = ' '.join([word] * n_words_in_line)
            f.write(line_to_use + '\n')
            f.write(line_to_use + '\n')


if __name__ == '__main__':
    words_list = get_unique_word_list('data/train/movie_lines_cleaned_10k.txt')
    generate_repeated_phrases('data/train/repeated_phrases.txt', words_list)
    generate_repeated_phrases('data/test/repeated_phrases.txt', words_list, n_lines=50)