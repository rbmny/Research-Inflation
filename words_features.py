import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open('../Databases/Data/Matrices/Forest/xgboost_predict_matrix_5k.pickle', 'rb') as handle:
    matrix1 = pickle.load(handle)
    matrix1 = pd.DataFrame(columns=matrix1.keys(), data=matrix1).T


def replace(x):
    return 1 if x >0 else 0


matrix1 = matrix1.map(replace)


# matrix1['words'] = matrix1.apply(
#     lambda row: sorted([(i, np.where(row['importances'] == i)[0][0]) for i in row['importances'] if i > 0],
#                        reverse=True), axis=1)
#
# with open('../Databases/Data/Globals/5k.pickle', 'rb') as handle:
#     count = pickle.load(handle)
#
# PERCENT = .9
#
# matrix1['dict'] = matrix1.apply(lambda row: [count[i[1]] for i in row.words], axis=1)
# matrix1['vectors'] = matrix1.apply(lambda row: [i[1] for i in row.words], axis=1)
# matrix1['sum'] = matrix1.apply(lambda row: sum(row.importances), axis=1)
# matrix1['len'] = matrix1.apply(lambda row: len(row.dict), axis=1)
# matrix1['pre'] = matrix1.apply(lambda row: [i[0] for i in row['words']], axis=1)
# matrix1['cumsum'] = matrix1.apply(lambda row: np.cumsum(row['pre']), axis=1)
# matrix1['idx'] = matrix1.apply(lambda row: np.argmin(np.abs(row['cumsum']-0.9)), axis=1)
# # bow = set([k for i in matrix1['vectors'].tolist() for k in i])


with open('../Databases/Data/Matrices/Argmin/xgboost2.pickle', 'wb') as handle:
    pickle.dump(matrix1, handle, protocol=pickle.HIGHEST_PROTOCOL)