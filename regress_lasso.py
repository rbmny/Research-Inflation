import os
import pickle
import pandas as pd
import decimal
import numpy as np

from joblib import parallel_backend
from numpy import arange
# import sklearn
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import multiprocessing as mp

from sklearn.preprocessing import StandardScaler

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)

with open('../Databases/Data/Matrices/matrix_epu.pickle', 'rb') as f:
    w2v_matrix = pickle.load(f)

# matrix2 = w2v_matrix.groupby(pd.Grouper(key='date',freq='ME')).sum()
# matrix2.drop(matrix2.tail(2).index,inplace=True)
#
temp = inflation_deltas['delta_1'].shift(-1).tolist()
#
# matrix2['predict'] = temp[:-1]
#
# data = matrix2.values
# X, y = data[:, :-1], data[:, -1]


# row = matrix2.iloc[48][:-1].values
# yhat = model.predict([row])
# difference = (yhat-matrix2.iloc[48, -1])**2
# row = matrix2.iloc[49][:-1].values
# yhat1 = model.predict([row])
# difference1 = (yhat - matrix2.iloc[49, -1]) ** 2
# row = matrix2.iloc[50][:-1].values
# yhat2 = model.predict([row])
# difference2 = (yhat - matrix2.iloc[50, -1]) ** 2
# row = matrix2.iloc[51][:-1].values
# yhat3 = model.predict([row])
# difference3 = (yhat - matrix2.iloc[51, -1]) ** 2
# print("alpha:",i, yhat, yhat1, yhat2, yhat3, difference, difference1, difference2, difference3)


path = "/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/Matrices"
dir_list = os.listdir(path)
dir_list = [f for f in dir_list if os.path.isfile(path + '/' + f)]
dir_list.remove(".DS_Store")
dir_list = ['matrix_15k.pickle']


def lasso_matrix(e):
    with open(f'{path}/{e}', 'rb') as handle:
        df = pickle.load(handle)
        # df = df.rename(columns={df.columns[0]: 'date_time'})

        matrix2 = df.groupby(pd.Grouper(key='date', freq='ME')).sum()
        matrix2.drop(matrix2.tail(2).index, inplace=True)

        matrix2['predict'] = temp[:-1]
        df_predict = pd.DataFrame(columns=['alpha1', 'predict1', 'actual', 'coef'])

    print(f'file {e} loaded \nregression started')
    with parallel_backend('threading', n_jobs=2):
        data = matrix2.values
        # scaler = StandardScaler()
        # scaler.fit(data)
        # data = scaler.transform(data)

        for i in range(47, len(temp) - 2):
            print(f'file {e}: {(i - 47) / (len(temp) - 47) * 100}% complete')

            X, y = data[:i, :-1], temp[:i]

            # Manual Tuning
            model = Lasso()
            # define model evaluation method
            # define grid
            # define search
            model.fit(X, y)
            # summarize
            # print('MAE: %.3f' % results.best_score_)
            # print('Config: %s' % results.best_params_)
            alpha1 = model.alpha
            coef = model.coef_

            row = data[i + 1, :-1].reshape(1, -1)
            yhat = model.predict(row)

            actual = temp[i + 1]
            row = [alpha1, yhat, actual, coef]
            df_predict.loc[i] = row

    with open(f'{path}/Predict/predict_{e}', 'wb') as handle:
        pickle.dump(df_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'file{e} created\nregression completed')


def parallel_lasso(files, num_workers=mp.cpu_count()):
    with mp.Pool(num_workers) as pool:
        matrix = pool.map(lasso_matrix, files)


if __name__ == '__main__':
    parallel_lasso(dir_list)
