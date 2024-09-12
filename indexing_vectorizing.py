import gc
import multiprocessing
import pickle
from collections import Counter
import pandas as pd


with open('../Databases/Data/df_count_indices.pickle', 'rb') as file1:
    data = pickle.load(file1)
    count_df = pd.DataFrame(data)
    counts = count_df['word_counts'].tolist()
del(data)

with open('../Databases/Data/text_list.pickle', 'rb') as file2:
    text_list = pickle.load(file2)

# with open('../Databases/Data/word_count.pickle', 'rb') as file3:
#     word_count = pickle.load(file3)

def create_index(text_count):
    keys = text_count.keys()
    vocab, index = {}, 1
    vocab['<pad>'] = 0
    for key in keys:
        vocab[key] = index
        index += 1
    inverse_vocab = {index: token for token, index in vocab.items()}
    return vocab, inverse_vocab


def parallel_indexing(text_count, num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(num_workers) as pool:
        # Count occurrence of key in each text in parallel
        indexes = pool.map(create_index, text_count)
    return indexes

def create_vectors(text_indices):
    vocab = text_indices[0]
    tokens = text_indices[1]
    vector = [vocab[word] for word in tokens]
    return vector


def parallel_vectoring(text_indices, num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(num_workers) as pool:
        # Count occurrence of key in each text in parallel
        vectors = pool.map(create_vectors, text_indices)
    return vectors



if __name__ == '__main__':
    #
    # indices = parallel_indexing(counts)
    #
    # vocabs = [vocab[0] for vocab in indices]
    # inverse_vocabs = [vocab[1] for vocab in indices]
    #
    #
    # count_df['vocabs'] = vocabs
    # count_df['inverse_vocabs'] = inverse_vocabs
    #
    # with open('../Databases/Data/df_count_indices.pickle', 'wb') as file1:
    #     pickle.dump(count_df, file1)
    #
    # del (indices)
    # del(count_df)
    # file1.close()
    # file2.close()

    vocabs = count_df['remapped_dicts'].tolist()
    zipped = list(zip(vocabs, text_list))
    del(text_list, vocabs)
    gc.collect()

    # vocab = indices[0]
    # inverse_vocab = indices[1]
    # with open('../Databases/Data/vocab_dict.pickle', 'wb') as file2:
    #     pickle.dump(vocab, file2)
    # with open('../Databases/Data/inverse_vocab_dict.pickle', 'wb') as file2:
    #     pickle.dump(inverse_vocab, file2)
    # vocab_size = len(vocab)

    vectors = parallel_vectoring(zipped)
    with open('../Databases/Data/df_count_indices.pickle', 'wb') as file1:
        count_df['remapped_vectors'] = vectors
        pickle.dump(count_df, file1)
    file1.close()

    # vectors = parallel_vectoring(text_list)
    print(count_df)
