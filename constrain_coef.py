import pickle
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.metrics import mean_squared_error


def fit_constrained_linear_regression(X, y):
    # Define the variable for the coefficients
    coefs = cp.Variable(X.shape[1])

    # Define the objective: minimize the mean squared error
    objective = cp.Minimize(cp.sum_squares(X @ coefs - y))

    # Define the constraints: sum of coefficients equals 1
    constraints = [cp.sum(coefs) == 1]

    # Set up the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the optimized coefficients
    return coefs.value


def predict(X_test, coefs):
    return X_test @ coefs


with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)

inflation_deltas['delta_1'] = inflation_deltas['delta_1'].fillna(0)
temp = inflation_deltas['delta_1'].tolist()[1:]

with open('../Databases/Data/Matrices/matrix_15k.pickle', 'rb') as f:
    df = pickle.load(f)
df = df.rename(columns={df.columns[0]: 'date_time'})
matrix2 = df.groupby(pd.Grouper(key='date_time', freq='ME')).sum()
matrix2.drop(matrix2.tail(2).index, inplace=True)

data = matrix2.values

if __name__ == '__main__':
    df_predict = pd.DataFrame(columns=['date', 'predict', 'actual', 'mse', 'coef'])

    for i in range(47, len(temp) - 1):
        print(f'{(i - 47) / (len(temp) - 47) * 100}% complete')
        X, y = data[i-47:i, :-1], temp[i-47:i]
        coefs = fit_constrained_linear_regression(X, y)
        x = data[i+1:i+13, :-1]
        actual = temp[i+1:i+13]
        z = predict(x, coefs)
        mse = mean_squared_error(actual, z)
        row = [matrix2.index[i + 1], z, actual, mse, coefs]
        df_predict.loc[i] = row

    with open('../Databases/Data/Matrices/Coefs/Constraints/constraint_coefs_matrix_15k.pickle', 'wb') as f:
        pickle.dump(df_predict, f)
