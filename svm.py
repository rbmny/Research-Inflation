import os
import pickle
import numpy as np
import pandas as pd
from joblib import parallel_backend
import multiprocessing as mp
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)


inflation_deltas['delta_1'] = inflation_deltas['delta_1']
inflation_deltas['delta_1'] = inflation_deltas['delta_1'].fillna(0)
temp = inflation_deltas['delta_1'].tolist()[1:]

# matrix2['predict'] = temp[:-1]
#
# data = matrix2.values
# X, y = data[:, :-1], data[:, -1]


path = "../Databases/Data/Matrices"
dir_list = os.listdir(path)
dir_list = [f for f in dir_list if os.path.isfile(path+'/'+f)]
# dir_list.remove(".DS_Store")
dir_list.remove("matrix_15k.pickle")
dir_list = ['matrix_10k.pickle', 'matrix_5k.pickle']

def lasso_matrix(e):
    with open(f'{path}/{e}', 'rb') as handle:
        df = pickle.load(handle)
        df = df.rename(columns={df.columns[0]: 'date_time'})
        matrix2 = df.groupby(pd.Grouper(key='date_time', freq='ME')).sum()
        matrix2.drop(matrix2.tail(2).index, inplace=True)

        # matrix2['predict'] = temp[:-1]
        df_predict = pd.DataFrame(columns=['predict', 'actual', 'mse'])

    print(f'file {e} loaded \nregression started')
    with parallel_backend('threading', n_jobs=8):
        data = matrix2.values

        # # Standardizing
        # scaler = StandardScaler()
        # scaler.fit(data)
        # data = scaler.transform(data)

        for i in range(47, len(temp)-1):
            print(f'file {e}: {(i-47)/(len(temp)-47)*100}% complete')

            X, y = data[i-47:i, :-1], temp[i-47:i]

            best_model = MLPRegressor(alpha=.04, random_state=42, max_iter=5000, shuffle=True, solver='lbfgs')
            best_model.fit(X, y)

            row = data[i + 1:i+12, :-1]
            yhat = best_model.predict(row)
            actual = temp[i + 1:i+12]
            mse = mean_squared_error(actual, yhat)

            row = [yhat, actual, mse]
            df_predict.loc[i] = row

    with open(f'{path}/Svm/rs42_5kiter_nn_svm_{e}', 'wb') as handle:
        pickle.dump(df_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'file {e} created\nregression completed')


def parallel_lasso(files):
    for i in files:
        lasso_matrix(i)
if __name__ == '__main__':
    lasso_matrix('matrix_5k.pickle')
    # parallel_lasso(dir_list)
