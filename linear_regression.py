import os
import pickle

import numpy
import numpy as np
import pandas as pd
from joblib import parallel_backend
import multiprocessing as mp

from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)


inflation_deltas['delta_1'] = inflation_deltas['delta_1'].fillna(0)
inflation_deltas = inflation_deltas.groupby(pd.Grouper(freq='ME')).sum()
temp = inflation_deltas['delta_1'].tolist()[1:]


path = "/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/Matrices"
dir_list = os.listdir(path)
dir_list = [f for f in dir_list if os.path.isfile(path+'/'+f)]
dir_list.remove(".DS_Store")

def lasso_matrix(e):
    with open(f'{path}/{e}', 'rb') as handle:
        matrix2 = pickle.load(handle)
        # df = df.rename(columns={df.columns[0]: 'date_time'})
        # matrix2 = df.groupby(pd.Grouper(key='date_time', freq='ME')).sum()
        matrix2.drop(matrix2.tail(1).index, inplace=True)
        matrix2 = matrix2.merge(inflation_deltas['delta_1'], how='left', right_index=True, left_index=True)

        # matrix2['predict'] = temp[:-1]
        df_predict = pd.DataFrame(columns=['date', 'predict', 'actual', 'mse', 'coef', 'intercept', 'c1'])

    print(f'file {e} loaded \nregression started')
    with parallel_backend('threading', n_jobs=8):
        data = matrix2.values[:, :-1]
        x = matrix2.values[:, -1].reshape(-1, 1)

        # Standardizing
        # scaler = StandardScaler()
        # scaler.fit(data)
        # data = scaler.transform(data)
        data = numpy.append(data, x, axis=1)
        for i in range(47, len(temp)-1):
            print(f'file {e}: {(i-47)/(len(temp)-47)*100}% complete')

            X, y = data[:i, :-1], data[:i, -1]

            best_model = linear_model.LinearRegression()
            best_model.fit(X, y)

            # row = scaler.transform((data[i + 1:i+12, :-1]))
            row = data[i + 1:i + 12, :-1]
            yhat = best_model.predict(row)
            actual = data[i + 1:i+12, -1]
            mse = mean_squared_error(actual, yhat)

            row = [matrix2.index[i+1], yhat, actual, mse, best_model.coef_, best_model.intercept_, data[i+1, 0]]
            df_predict.loc[i] = row

    with open(f'{path}/Linear/L_predict_{e}', 'wb') as handle:
        pickle.dump(df_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'file {e} created\nregression completed')


def parallel_lasso(files, num_workers=mp.cpu_count()):
    with mp.Pool(num_workers) as pool:
        matrix = pool.map(lasso_matrix, files)

if __name__ == '__main__':
    # lasso_matrix('matrix_10k.pickle')
    lasso_matrix('matrix_ratios_5.pickle')
    # parallel_lasso(dir_list)
    print()
