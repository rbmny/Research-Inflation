from datetime import datetime
from dateutil.relativedelta import relativedelta
from gensim.models import Word2Vec
from multiprocessing import cpu_count, Pool, Manager
import pandas as pd
import pickle

# Load the pickle data
with open("../Databases/Processed/NYT_agg_processed_texts.pickle", "rb") as file1:
    agg_df = pickle.load(file1)

# Convert('date') field to datetime type if it's not and sort the DataFrame by date.
agg_df['date'] = pd.to_datetime(agg_df['date'])
agg_df = agg_df.sort_values('date')

# Define the start and end date for the rolling window
start_date = agg_df['date'].min()
end_date = datetime(year=2024, month=6, day=30) - relativedelta(years=4)  # To ensure at least 4 years of data


# Define a worker function
def train_model(args):
    current_date, progress_dict = args  # unpacking tuple of arguments

    four_year_data = agg_df[(agg_df['date'] >= current_date) & (agg_df['date'] <= current_date + relativedelta(years=4))]
    processed_text = four_year_data['processed_text'].tolist()  # Assuming 'processed_text' is the text column

    # If your data is not already tokenized
    processed_text = [text.split() for text in processed_text]

    # Train a Word2Vec model
    model = Word2Vec(sentences=processed_text, vector_size=200, window=5, workers=6, min_count=50)

    model.save(
        f"Models/Rolling/word2vec_{current_date.strftime('%Y%m')}_to_{(current_date + relativedelta(years=4)).strftime('%Y%m')}.model")

    # Update progress in dictionary
    progress_dict[current_date.strftime('%Y%m')] = True

    # Compute and print progress
    num_completed = sum(progress_dict.values())
    total = len(progress_dict)
    print(f"Progress: {num_completed / total * 100: .2f}% completed")


def main():
    with Manager() as manager:
        # Define a pool of workers equal to the number of CPUs
        pool = Pool(cpu_count())

        # A Manager dict to keep track of the progress
        progress_dict = manager.dict()

        # Dates for the rolling window
        dates = pd.date_range(start_date, end_date, freq='ME')

        # Add False entry to dictionary for each date
        for date in dates:
            progress_dict[date.strftime('%Y%m')] = False

        # Pack progress_dict with each date
        args = [(date, progress_dict) for date in dates]

        # Train a Word2Vec model for every starting month in parallel
        pool.map(train_model, args)

        print('Models saved.')


if __name__ == '__main__':
    main()


# #
# #
# # Train a bigram detector.
# bigram_transformer = Phrases(processed_text)
# bigram_model = Word2Vec(bigram_transformer[processed_text], min_count=10, workers=6)
# bigram_model.save("Models/bigram_10.model")
#
#
# # Trigram training
# # bigram_transformer = Phrases(processed_text)
# trigram_transformer = Phrases(bigram_transformer[processed_text])
# trigram_model = Word2Vec(trigram_transformer[bigram_transformer[processed_text]], min_count=10, workers=6)

# ???
# for sent in processed_text:
#     bigrams_ = [b for b in bigram_transformer[sent] if b.count('_') == 1]
#     trigrams_ = [t for t in trigram_transformer[bigram_transformer[sent]] if t.count('_') == 2]

# trigram_model.save("Models/trigram_10.model")