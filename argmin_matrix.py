import gc
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp

with open('../Databases/Data/Matrices/Argmin/xgboost.pickle', 'rb') as f:
    wordlist = pickle.load(f)
    wordlist = wordlist.apply(lambda row: row['dict'][:row['idx']], axis=1)
    wordlist = wordlist.tolist()


with open('../Databases/Data/df_count.pickle', 'rb') as f:
    df_count = pickle.load(f)


def count_words_in_text(tokens):
    row_vec = []
    for i in tokens[1]:
        row_vec.append(tokens[0][i])
    return row_vec


# Function to count words in multiple texts using parallel processing
def parallel_count_words(counters, related_words, num_workers=mp.cpu_count()):
    with mp.Pool(num_workers) as pool:
        matrix = pool.map(count_words_in_text, zip(counters, related_words))
    return matrix


if __name__ == '__main__':
    word_set = set([j for i in wordlist for j in i])
    np_matrix = parallel_count_words(df_count['word_counts'].tolist(), related_words=wordlist)

    dates = df_count['date'].tolist()
    for index, i in enumerate(dates):
        np_matrix[index].insert(0, i)

    np_matrix = np.array(np_matrix)
    df2 = pd.DataFrame(np_matrix, columns=['date', *word_set])

    # memory management

    with open(f'../Databases/Data/Matrices/A_min_counts/changing_words.pickle', 'wb') as handle:
        pickle.dump(df2, handle)