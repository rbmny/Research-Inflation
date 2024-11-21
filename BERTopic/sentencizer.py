from nltk.tokenize import sent_tokenize
import pickle
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def tokenize_text(text):
    """Tokenize text into sentences."""
    return sent_tokenize(text)


def parallel_process_entries(df):
    """Process DataFrame entries in parallel."""
    dates = df['date'].values
    texts = df['text'].values

    # Using multiprocessing Pool to parallelize tokenization
    with Pool(cpu_count()) as pool:
        all_sentences = list(tqdm(pool.imap(tokenize_text, texts), total=len(texts), desc="Tokenizing sentences"))

    # Flatten the list of lists
    sentences = [sentence for sublist in all_sentences for sentence in sublist]
    result_dates = [date for idx, date in enumerate(dates) for _ in range(len(all_sentences[idx]))]

    # Create the DataFrame
    sentence_df = pd.DataFrame({'date': result_dates, 'sentence': sentences})
    return sentence_df


if __name__ == '__main__':
    # Load the WSJ database
    with open("../Database/WSJ_database.pickle", "rb") as f:
        wsj_df = pickle.load(f)

    # Process the wsj_df to get the new sentence-based DataFrame
    sentence_df = parallel_process_entries(wsj_df)

    # Save the new DataFrame to check results
    sentence_df.to_pickle("../Database/WSJ_sentences_database.pickle")

    # Remove original 'text' entries from wsj_df
    wsj_df.drop(columns=['text'], inplace=True)


