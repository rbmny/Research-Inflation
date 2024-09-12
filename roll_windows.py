import pickle
import pandas as pd
import os


# Get the list of all files and directories
path = "/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/Matrices"
dir_list = os.listdir(path)
dir_list = [f for f in dir_list if os.path.isfile(path+'/'+f)]
dir_list.remove(".DS_Store")
window = 30
for i in dir_list:
    with open(f'{path}/{i}', 'rb') as handle:
        df = pickle.load(handle)
    df = df.set_index('date')
    df_rolled = df.rolling(window, min_periods=1).sum()
    with open(f'{path}/rolled_{window}_{i}', 'wb') as handle:
        pickle.dump(df_rolled, handle, protocol=pickle.HIGHEST_PROTOCOL)
