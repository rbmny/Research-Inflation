import pandas as pd
import gc
import numpy as np
import multiprocessing
import pickle
from collections import Counter



# open file and save to df
with open('../Databases/Processed/NYT_agg_processed_texts.pickle', 'rb') as file1:
    data = pickle.load(file1)
    df = pd.DataFrame(data)

with open('../Databases/Data/df_count.pickle', 'rb') as file2:
    data = pickle.load(file2)
    df2 = pd.DataFrame(data)


def word_combinations(text_elements):
    two_words = ['_'.join(ws) for ws in zip(text_elements, text_elements[1:])]
    three_words = ['_'.join(ws) for ws in zip(text_elements, text_elements[1:], text_elements[2:])]
    return Counter(three_words)

    # IMPORTANT make dictionary for most common words
    # wordscount = {w: f for w, f in Counter(two_words).most_common()}
    # return wordscount

def parallel_count(text_elements_param, num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(num_workers) as pool:
        # Count occurrence of key in each text in parallel
        counts = pool.map(word_combinations, text_elements_param)

    # comment line for total counts
    return counts

    total_counts = Counter()
    for count in counts:
        total_counts.update(count)

    return total_counts


if __name__ == '__main__':
    dates = df['date'].tolist()

    # Split the tokens column strings into lists of words
    df['processed_text'] = df['processed_text'].apply(lambda x: x.split())

    text_elements = df['processed_text'].tolist()

    # for index, row in df.iterrows():
    #     columns = row['processed_text']
    #     text = df.loc[index, 'processed_text']
    #     text_elements.append([text, columns])

    del df, df2
    gc.collect()

    matrices = parallel_count(text_elements)
    # total_count = parallel_count(text_elements)
    # with open('../Databases/Data/Counts/trigram_10_counts.pickle', 'wb') as file1:
    #     pickle.dump(total_count, file1)

    #
    df = pd.DataFrame(columns=['date', 'trigram_count'])
    df['date'] = dates
    df['trigram_count'] = matrices

    with open('../Databases/Data/Counts/trigram_10_df.pickle', 'wb') as file1:
        pickle.dump(df, file1)