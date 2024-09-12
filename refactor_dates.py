import pickle
from gensim.models import Phrases
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import Cython
import pandas as pd


df = pd.read_csv('/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/Inflation/CPIAUCSL.csv')
df.rename(columns={'DATE': 'date'}, inplace=True)
df = df.set_index('date')
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

# matrix1 = matrix1.set_index('date')
# matrix2 = matrix1.groupby(pd.Grouper(freq='ME')).sum()

start_remove = pd.to_datetime('1947-01-01 00:00:00')
end_remove = pd.to_datetime('2010-01-01 00:00:00')
df = df.loc[(df.index < start_remove) | (df.index >= end_remove)]

for i in range(1,13):
    df[f'delta_{i}'] = ( df['CPIAUCSL'].shift(-i) - df['CPIAUCSL']) / df['CPIAUCSL'] *100
    df[f'delta_{i}'] = df[f'delta_{i}'].shift(i)

df.at['2010-01-01', 'delta_1'] = 0
#
with open('/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/Inflation/inflation_deltas.pkl', 'wb') as f:
    pickle.dump(df, f)
print()

