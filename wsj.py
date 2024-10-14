import pickle
import pandas as pd
from numba import jit
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch


print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# For demonstration, let's create a sample DataFrame
with open("./Database/WSJ_database.pickle", "rb") as f:
    wsj_df = pickle.load(f)

wsj_df = wsj_df[['date', 'text']]
wsj_df['text'] = wsj_df['text'].apply(lambda x: x[:512])

# Initialize the pipeline for text classification
pipe = pipeline("text-classification", model="Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier",
                device=0 if torch.cuda.is_available() else -1)


class WSJDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx]['text']


# Initialize the dataset
dataset = WSJDataset(wsj_df)

# Initialize lists to store the results
classification_names = []
values = []


def load_batch(batch_size):
# for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")

    # Initialize DataLoader with the specified batch size
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Create a TQDM progress bar
    pbar = tqdm(total=len(dataset))

    for batch in dataloader:
        out = pipe(batch, batch_size=batch_size)
        for o in out:
            classification_names.append(o['label'])
            values.append(o['score'])
        pbar.update(len(batch))

    pbar.close()

    # Add the classification results to the DataFrame
    wsj_df['classification_name'] = classification_names
    wsj_df['value'] = values

    # Save the updated DataFrame to a new file
    output_file = "./Database/WSJ_database_classified.csv"
    wsj_df.to_csv(output_file, index=False)

    print(f"Classification results saved to {output_file}")

load_batch(64)