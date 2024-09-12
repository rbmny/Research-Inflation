import pickle

import numpy as np
import pandas as pd
import multiprocessing
from collections import Counter

# open a file, where you stored the pickled data
file = open('../Databases/Processed/NYT_agg_processed_texts.pickle', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()


# Function to count words in a single preprocessed text
def count_words_in_text(tokens):
    return Counter(tokens)


# Function to count words in multiple texts using parallel processing
def parallel_count_words(preprocessed_texts, num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(num_workers) as pool:
        # Count words in each text in parallel
        counts = pool.map(count_words_in_text, preprocessed_texts)
    for count in counts:
            count = Counter(count)
    return counts
    # Combine the counts from all texts
    total_counts = Counter()
    for count in counts:
        total_counts.update(count)

    return total_counts


# Example usage
if __name__ == "__main__":

    df = pd.DataFrame(data)

    # Split the tokens column strings into lists of words
    df['processed_text'] = df['processed_text'].apply(lambda x: x.split())



    # Ensure that the tokens column is now a Series of lists
    if isinstance(df['processed_text'], pd.Series) and all(isinstance(i, list) for i in df['processed_text']):
        # Extract the list of tokens from the DataFrame
        preprocessed_texts = df['processed_text'].tolist()

        # Count words
        df['word_counts'] = parallel_count_words(preprocessed_texts)
        # w_list = parallel_count_words(preprocessed_texts).keys()


        with open('../Databases/Data/df_count.pickle', 'wb') as f:
            pickle.dump(df, f)
            f.close()
        print(df)
        # with open('../Databases/Data/vocabulary_list.pickle', 'wb') as file:
        #     # A new file will be created
        #     pickle.dump(list(w_list), file)
        #
        # with open('../Databases/Data/text_list.pickle', 'wb') as file2:
        #     pickle.dump(preprocessed_texts, file2)

    else:
        print("The tokens column must be a Series of lists.")