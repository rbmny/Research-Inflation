import os
import pickle
import numpy as np
import pandas as pd
from joblib import parallel_backend
import multiprocessing as mp

from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import tree
import xgboost as xgb


from sklearn.preprocessing import StandardScaler

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)


inflation_deltas['delta_1'] = inflation_deltas['delta_1'].fillna(0)
inflation_deltas = inflation_deltas.groupby(pd.Grouper(freq='ME')).sum()
temp = inflation_deltas['delta_1'].tolist()[1:]
window = 47
eval_size = 6

# matrix2['predict'] = temp[:-1]
#
# data = matrix2.values
# X, y = data[:, :-1], data[:, -1]


path = "../Databases/Data/Matrices"
dir_list = os.listdir(path)
dir_list = [f for f in dir_list if os.path.isfile(path+'/'+f)]
# dir_list.remove(".DS_Store")
dir_list.remove("matrix_15k.pickle")
dir_list.remove("matrix_10k.pickle")

def lasso_matrix(e):
    with open(f'{path}/{e}', 'rb') as handle:
        df = pickle.load(handle)
        df = df.rename(columns={df.columns[0]: 'date_time'})
        matrix2 = df.groupby(pd.Grouper(key='date_time', freq='ME')).sum()
        matrix2.drop(matrix2.tail(1).index, inplace=True)
        matrix2 = matrix2.merge(inflation_deltas['delta_1'], how='left', right_index=True, left_index=True)

        # matrix2['predict'] = temp[:-1]
        df_predict = pd.DataFrame(columns=['predict', 'actual', 'mse', 'importances'])

    print(f'file {e} loaded \nregression started')
    with (parallel_backend('threading', n_jobs=8)):
        data = matrix2.values[:, :-1]
        x = matrix2.values[:, -1].reshape(-1, 1)

        # Standardizing
        # scaler = StandardScaler()
        # scaler.fit(data)
        # data = scaler.transform(data)
        data = np.append(data, x, axis=1)
        window_importance = {}
        for i in range(window, len(temp)-1):
            print(f'file {e}: {(i-window)/(len(temp)-window)*100}% complete')

            X, y = data[i-window:i, :-1], data[i-window:i, -1]

            X_train, y_train = X[:-eval_size], y[:-eval_size]
            X_eval, y_eval = X[-eval_size:], y[-eval_size:]

            # best_model = xgb.XGBRegressor(objective="reg:squarederror",feature_selector='cyclic', early_stopping_rounds=50, booster='gblinear')
            # best_model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=1,  early_stopping_rounds=50, subsample=1, colsample_bynode=1,)
            g_depth = [1,3,6] # default 1
            g_lr = [0.05, 0.1, .15, .2, .25, .3, .35, .4] # default .3
            grid = [(x, y) for x in g_depth for y in g_lr]
            # grid = pd.DataFrame(columns=g_depth, index=g_lr, )
            global idx_grid
            idx_grid = 0
            best_score = 9999999
            for depth, lr in grid:
                best_model = xgb.XGBRegressor(objective="reg:squarederror", max_depth=int(depth), early_stopping_rounds=50,
                                              subsample=1, colsample_bynode=1, learning_rate=lr)
                best_model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)])
                if best_model.best_score < best_score:
                    idx_grid = grid.index((depth, lr))
                    window_importance[i] = best_model.feature_importances_

    with open(f'{path}/Forest/xgboost_predict_{e}', 'wb') as handle:
        pickle.dump(window_importance, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'file {e} created\nregression completed')


def parallel_lasso(files, num_workers=mp.cpu_count()):
    with mp.Pool(num_workers) as pool:
        matrix = pool.map(lasso_matrix, files)

if __name__ == '__main__':
    lasso_matrix('matrix_5k.pickle')

    # parallel_lasso(dir_list)
