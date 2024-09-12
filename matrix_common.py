import multiprocessing
import pickle
import pandas as pd

# open files
file1 = open('../Databases/Data/df_count.pickle', 'rb')
file2 = open('../Databases/Data/Globals/word_count.pickle', 'rb')

# load processed texts and dates into df
data = pickle.load(file1)
df = pd.DataFrame(data)

# load total word count counter
wcntr = pickle.load(file2)


# Get list of most common words
tokens = [i[0] for i in wcntr.most_common(5000) if i[1] >= 50]

with open('../Databases/Data/Globals/5k.pickle', 'wb') as file:
    # A new file will be created
    wdict = {k: tokens[k] for k in range(len(tokens))}
    reverse_dict = {tokens[k]: k for k in range(len(tokens))}
    pickle.dump(tokens, file)
    print('dumped')


# count occurrences of tokens for each date
def count_words(counter):
    col_vectors = []
    for key in tokens:
        col_vectors.append(counter[key])
    return col_vectors


# use parallel computing for counting tokens
def parallel_count(counter_list, num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(num_workers) as pool:
        # Count occurrence of key in each text in parallel
        counts = pool.map(count_words, counter_list)
    return counts


if __name__ == '__main__':

    # Initialize matrix with common words + date as columns
    columns = ["date_time", *tokens]

    # Convert counters to list for parallel processing
    counter_list = df['word_counts'].tolist()
    row_vectors = parallel_count(counter_list)

    # Populate add date to row vectors
    for index, row in df.iterrows():
        row_vectors[index].insert(0, row['date'])
    # print(row_vectors)

    # Populate matrix based on row vectors with dates
    matrix = pd.DataFrame(columns=columns, data=row_vectors)
    matrix = matrix.rename(columns={matrix.columns[0]: 'date_time'})
    # matrix2 = matrix.groupby(pd.Grouper(key='date_time', freq='ME')).sum()

    with open('../Databases/Data/Matrices/matrix_10k.pickle', 'wb') as file:
        # A new file will be created
        pickle.dump(matrix, file)

# close the files
file1.close()
file2.close()
