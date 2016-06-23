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


def create_dataset(output_filename, words_list, n_lines=10000, n_words_in_line=3, same_words=False, reversed_output=False):
    with open(output_filename, 'w') as f:
        for i in xrange(n_lines):
            if same_words:
                words = [np.random.choice(words_list)] * n_words_in_line
            else:
                words = np.random.choice(words_list, size=n_words_in_line)

            line_to_use = ' '.join(words) + '\n'
            f.write(line_to_use)
            if reversed_output:
                line_to_use = ' '.join(reversed(words)) + '\n'
            f.write(line_to_use)


if __name__ == '__main__':
    words_list = get_unique_word_list('data/train/movie_lines_cleaned_10k.txt')
    create_dataset('data/train/repeated_words.txt', words_list, same_words=True)
    create_dataset('data/test/repeated_words.txt', words_list, same_words=True, n_lines=50)
    create_dataset('data/train/repeated_phrases.txt', words_list)
    create_dataset('data/test/repeated_phrases.txt', words_list, n_lines=50)
    create_dataset('data/train/reversed_phrases.txt', words_list, reversed_output=True)
    create_dataset('data/test/reversed_phrases.txt', words_list, n_lines=50, reversed_output=True)