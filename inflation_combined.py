import pickle
import pandas as pd
import multiprocessing as mp

with open('../Databases/Data/Inflation/inflat_df.pkl', 'rb') as f:
    df = pickle.load(f)

columns = df.columns
price_col = set([int(i.removeprefix('price_index_series_')) for i in columns if 'price' in i])
weight_col = set([int(i.removeprefix('weight_series_')) for i in columns if 'weight' in i])
# price_col.sort()
# weight_col.sort()

new_df = pd.DataFrame()
things = []
error = []
df['sum'] = 0
df.fillna(0, inplace=True)
for i in price_col:
    try:
        df['sum'] += df.apply(lambda row:
                              (+row[f'price_index_series_{i}'] * row[f'weight_series_{i}']),
                                    axis=1)
    except:
        error.append(i)

print()
# df_weights
