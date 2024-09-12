import pickle
import pandas as pd

# open a file, where you stored the pickled data
with open('../Databases/Raw/WSJ_database.pickle', 'rb') as file:
    # dump information to that file
    wsj = pickle.load(file)
    wsj = wsj[['date', 'text']]

with open('../Databases/Raw/NYT_database.pickle', 'rb') as file:
    # dump information to that file
    nyt = pickle.load(file)
    nyt = nyt[['date', 'text']]

with open('../Databases/Raw/REUTERS_database.pickle', 'rb') as file:
    # dump information to that file
    reut = pickle.load(file)
    reut = reut[['date', 'text']]

df_total = pd.concat([wsj, nyt, reut], axis=1)
print()