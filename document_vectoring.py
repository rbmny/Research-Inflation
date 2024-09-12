import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import gensim
from dateutil.relativedelta import relativedelta
from gensim.models import Word2Vec
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)


def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


def mbkmeans_clusters(
        X,
        k,
        mb,
        print_silhouette_values,
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")
    return km, km.labels_


if __name__ == '__main__':

    with open('../Databases/Processed/NYT_agg_processed_texts.pickle', 'rb') as f:
        agg_df = pickle.load(f)

    # Define the start and end date for the rolling window
    start_date = agg_df['date'].min()
    end_date = datetime(year=2024, month=6, day=30) - relativedelta(years=4) # To ensure at least 4 years of data

    dates = pd.date_range(start_date, end_date, freq='ME')

    for date in dates:
        print(dates.get_loc(date)/len(dates),'%')
        current_date = date

        # Construct the model filename based on the current date
        model_filename = f"Models/Rolling/word2vec_{current_date.strftime('%Y%m')}_to_{(current_date + relativedelta(years=4)).strftime('%Y%m')}.model"

        if os.path.exists(model_filename):
            # Load the model from the file
            model = Word2Vec.load(model_filename)

        data_for_model = agg_df[
            (agg_df['date'] >= current_date) & (agg_df['date'] <= current_date + relativedelta(years=4))]

        tokenized_docs = [doc.split() for doc in data_for_model['processed_text'].tolist()]

        vectorized_docs = vectorize(tokenized_docs, model=model)
        kmax = 300
        k_clust = 13

        clustering, cluster_labels = mbkmeans_clusters(
            X=vectorized_docs,
            k=k_clust,
            mb=5306,
            print_silhouette_values=True,
        )
        df_clusters = pd.DataFrame({
            "date": data_for_model['date'],
            "text": tokenized_docs,
            "tokens": [" ".join(text) for text in tokenized_docs],
            "cluster": cluster_labels
        })
        with open(f"../Databases/Data/Matrices/Clusters/Rolling/{k_clust}clusters_{current_date.strftime('%Y%m')}_to_{(current_date + relativedelta(years=4)).strftime('%Y%m')}.pickle", 'wb') as f:
            pickle.dump(df_clusters, f)