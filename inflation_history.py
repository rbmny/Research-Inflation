import gc

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import multiprocessing as mp

history = pd.read_csv('../Databases/Data/Inflation/historico_nsa.csv', sep=';')


entities = sorted(history['entity_name'].unique())
dates = sorted(history['date'].unique())


df = pd.DataFrame(index=dates, columns=entities)

del(entities)
gc.collect()


def populate(date):
    y = history.query(f'date == "{date}"')
    zipped = dict(zip(y['entity_name'].tolist(), y['value'].tolist()))
    return pd.Series(zipped, name=date)


def update_df(result_df):
    global df
    df.update(result_df)


def parallel_populate(dates, num_workers=mp.cpu_count()):
    with mp.Pool(num_workers) as pool:
        results = pool.map(populate, dates)

    # Concatenate the results into a DataFrame
    result_df = pd.concat(results, axis=1).T

    # Update the main DataFrame with the results
    update_df(result_df)


if __name__ == '__main__':
    parallel_populate(dates)
    with open('../Databases/Data/Inflation/inflat_df.pkl', 'wb') as f:
        pickle.dump(df, f)