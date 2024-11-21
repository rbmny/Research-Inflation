import pickle
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm
import os
import torch
from transformers import BertTokenizer, BertModel, logging as transformers_logging

# Suppress warnings from transformers which will print to console
transformers_logging.set_verbosity_error()

# Ensure the input file exists and load WSJ sentences database
input_path = "../Database/WSJ_sentences_database.pickle"
encode_path = "../Database/embeddings.pickle"
temp_output_path = "../Database/wsj_topics_keywords_temp.pkl"
final_output_path = "../Database/wsj_topics_keywords.pickle"
if not os.path.exists(input_path):
    raise FileNotFoundError(f"The file {input_path} does not exist.")

with open(input_path, "rb") as f:
    wsj_df = pickle.load(f)

# Check for necessary columns
if 'date' not in wsj_df.columns or 'sentence' not in wsj_df.columns:
    raise ValueError("The input DataFrame must contain 'date' and 'sentence' columns.")

# Load pre-trained BERT model and tokenizer from transformers
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Check if CUDA is available and use the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Instantiate BERTopic with CPU-based UMAP and HDBSCAN
from hdbscan import HDBSCAN
from umap import UMAP

topic_model = BERTopic(
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom',
                          prediction_data=True),
    calculate_probabilities=True, verbose=True
)


# Define a generator to stream sentences from the DataFrame
def sentence_generator(df, batch_size=1000):  # Ensure batch_size is consistent
    for start in range(0, len(df), batch_size):  # Start from 0
        end = min(start + batch_size, len(df))
        yield df['sentence'].iloc[start:end].tolist(), df['date'].iloc[start:end].tolist()


# Function to extract keywords for a topic
def get_keywords(topic_model, topic):
    keywords = topic_model.get_topic(topic)
    return ', '.join([word for word, _ in keywords])


# Function to append a DataFrame to a pickle file
def append_to_pickle_file(df, file_path):
    with open(file_path, "ab") as f:
        pickle.dump(df, f, protocol=4)


# Function to read and combine data from a pickle file
def read_combined_pickle_file(file_path):
    dataframes = []
    with open(file_path, "rb") as f:
        while True:
            try:
                dataframes.append(pickle.load(f))
            except EOFError:
                break
    return pd.concat(dataframes, ignore_index=True)


# Calculate the total number of batches for tqdm progress display
batch_size = 1000  # Consistent batch size
total_sentences = len(wsj_df)
total_batches = (total_sentences + batch_size - 1) // batch_size

# Process data in streaming fashion and append to temp pickle file
print("Processing data in streaming fashion...")

for batch_sentences, batch_dates in tqdm(
        sentence_generator(wsj_df, batch_size=batch_size),
        desc="Processing batches",
        total=total_batches):
    batch_topics, batch_probs = topic_model.fit_transform(batch_sentences)

    # Extract keywords
    batch_keywords = [get_keywords(topic_model, topic) if topic != -1 else "" for topic in batch_topics]

    # Create a batch DataFrame
    batch_df = pd.DataFrame({
        'date': batch_dates,
        'text': batch_sentences,
        'topic': batch_topics,
        'keywords': batch_keywords
    })

    # Append batch DataFrame to temp pickle file
    append_to_pickle_file(batch_df, temp_output_path)

# Read the temporary pickle file to get the combined DataFrame
new_df = read_combined_pickle_file(temp_output_path)

# Save the new DataFrame to the final pickle file
with open(final_output_path, "wb") as final_output_file:
    pickle.dump(new_df, final_output_file, protocol=4)

# Remove the temporary pickle file
os.remove(temp_output_path)

print(f"DataFrame saved to {final_output_path}")
