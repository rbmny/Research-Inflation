import os
import pickle
from datetime import datetime

import numpy
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
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
inflation_deltas = inflation_deltas[['delta_1']]
inflation_deltas = inflation_deltas.groupby(pd.Grouper(freq='ME')).sum()
agg_df = inflation_deltas.index

path = "/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/Matrices"


def lasso_matrix(e):
    # matrix2['predict'] = temp[:-1]
    df_predict = pd.DataFrame(columns=['predict', 'actual', ])

    print(f'regression started')
    with parallel_backend('threading', n_jobs=8):

        # Define the start and end date for the rolling window
        start_date = agg_df.min()
        end_date = datetime(year=2024, month=6, day=30) - relativedelta(years=4)  # To ensure at least 4 years of data

        dates = pd.date_range(start_date, end_date, freq='MS')
        for current_date in dates:

            with open(
                    f"../Databases/Data/Matrices/Rolling_Clusters/matrix_ratios_{current_date.strftime('%Y%m')}_to_{(current_date + relativedelta(years=4)).strftime('%Y%m')}.pickle",
                    'rb') as f:
                data3 = pickle.load(f)
                data_internal = pd.merge(data3, inflation_deltas, left_index=True, right_index=True, how='left')

                data2 = data_internal.values

                # mask = (data_internal.index >= current_date) & (
                #         data_internal.index <= current_date + relativedelta(months=len(data2)))

                # true_indices = numpy.where(mask == True)
                # mask_predict = max(true_indices[0])+1

                X, y = data2[:-1, :-1], data2[:-1, -1]
                if current_date == start_date:
                    X, y = X[1:], y[1:]

            best_model = linear_model.LinearRegression()
            best_model.fit(X, y)

            # row = scaler.transform((data[i + 1:i+12, :-1]))
            row = data2[-1, :-1]
            yhat = best_model.predict(row.reshape(1, -1))
            actual = data2[-1, -1]

            row = [yhat, actual]
            df_predict.loc[current_date] = row


    with open(f'{path}/Linear/cluster_treg.pickle', 'wb') as handle:
        pickle.dump(df_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'file {e} created\nregression completed')


def parallel_lasso(files, num_workers=mp.cpu_count()):
    with mp.Pool(num_workers) as pool:
        matrix = pool.map(lasso_matrix, files)


if __name__ == '__main__':
    # lasso_matrix('matrix_10k.pickle')
    lasso_matrix('matrix_5k.pickle')
    # parallel_lasso(dir_list)
