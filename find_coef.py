import pickle

import pandas as pd

with (open('../Databases/Data/Matrices/matrix_5k.pickle', 'rb') as f):
    df_count = pickle.load(f)
    df_count = df_count.rename(columns={df_count.columns[0]: 'date_time'})
    df_count = df_count.groupby(pd.Grouper(key='date_time', freq='ME')).sum()


with open('../Databases/Data/Matrices/Elastic/PREVISAO2matrix_5k.pickle', 'rb') as f:
    df_coef = pickle.load(f)

def mmultiply(words, coefs):
    row = [w * b for w, b in zip(words, coefs)]
    idx, minr, mar = row.index(min(row)), min(row), max(abs(max(row)), abs(min(row)))
    return row, idx, minr, mar

cnt1 = df_count.values[51, :]
cnt2 = df_count.values[53, :]
coef1 = df_coef.values[50, 6]
coef2 = df_coef.values[52, 6]

r1, idx1, min1, mar1 = mmultiply(df_count.values[51, :], df_coef.values[50, 6])
r2, idx2, min2, mar2 = mmultiply(df_count.values[53, :], df_coef.values[52, 6])
print()