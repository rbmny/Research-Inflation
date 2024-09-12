import os
import pickle
import numpy as np
import pandas as pd
from joblib import parallel_backend
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)


inflation_deltas['delta_1'] = inflation_deltas['delta_1']
inflation_deltas['delta_1'] = inflation_deltas['delta_1'].fillna(0)
temp = inflation_deltas['delta_1'].tolist()[1:]


# with open('../Databases/Data/Matrices/Elastic/e_predict_matrix_15k.pickle', 'rb') as f:
#     coef_file = pickle.load(f)
coef_dates = [i for i in range(47, 172)]
coef_dates.sort(reverse=True)

path = "/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/Matrices"
dir_list = os.listdir(path)
dir_list = [f for f in dir_list if os.path.isfile(path+'/'+f)]
dir_list.remove(".DS_Store")
# dir_list.remove("matrix_15k.pickle")
dir_list.remove("matrix_25k.pickle")

def predict(X_test, coefs):
    return X_test @ coefs

def lasso_matrix(e):
    with open(f'{path}/{e}', 'rb') as handle:
        df = pickle.load(handle)
        df = df.rename(columns={df.columns[0]: 'date_time'})
        matrix2 = df.groupby(pd.Grouper(key='date_time', freq='ME')).sum()
        matrix2.drop(matrix2.tail(2).index, inplace=True)



    print(f'file {e} loaded \nregression started')
    with parallel_backend('threading', n_jobs=8):
        data = matrix2.values

        with open(f'../Databases/Data/Matrices/Coefs/constraint_coefs_matrix_15k.pickle', 'rb') as f:
            coef_file = pickle.load(f)


        for coef_index in coef_dates:
            df_predict = pd.DataFrame(columns=['date', 'predict', 'actual', 'mse', 'coef'])
            frow = coef_file.loc[coef_index]


            for i in range(47, len(temp)-1):
                    print(f'file {coef_index} {e}: {(i-47)/(len(temp)-47)*100}% complete')

                    X, y = data[i-47:i, :-1], temp[i-47:i]
                    coef_ = frow['coef']


                    row = data[i + 1:i+12, :-1]
                    yhat = predict(row, coef_)
                    actual = temp[i + 1:i+12]
                    mse = mean_squared_error(actual, yhat)

                    row = [matrix2.index[i+1], yhat, actual, mse, coef_]
                    df_predict.loc[i] = row
            try:
                os.mkdir(f'{path}/Coefs/{e[:-7]}')
            except FileExistsError:
                print(f'Linear/Coefs/full_{e[:-7]}/{e}')
            with open(f'{path}/Coefs/{e[:-7]}/{coef_index}_{e}', 'wb') as handle:
                pickle.dump(df_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'file {coef_index} {e} created\nregression completed')


def parallel_lasso(files, num_workers=mp.cpu_count()):
    with mp.Pool(num_workers) as pool:
        matrix = pool.map(lasso_matrix, files)

if __name__ == '__main__':
    lasso_matrix('matrix_15k.pickle')
    # parallel_lasso(dir_list)
