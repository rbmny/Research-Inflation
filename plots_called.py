import plotly.express as px
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)
with open('../Databases/Data/Matrices/Elastic/e_predict_related_words.pickle', 'rb') as f:
    pmtx1 = pickle.load(f)
with open('../Databases/Data/Matrices/Elastic/adjusted_e_predict_matrix_5k.pickle', 'rb') as f:
    pmtx2 = pickle.load(f)
lmtx = [pmtx1, pmtx2]

dates = inflation_deltas.index.tolist()[48:]

def plot(pmtx, months):
    month_dif = 1 - months
    if month_dif != 0:
        pmtx = pmtx[:month_dif]
    pmtx.loc[:, 'p0'] = pmtx.predict.map(lambda x: x)
    pmtx.loc[:, 'a0'] = pmtx.actual.map(lambda x: x)

    pmtx['mse'] = pmtx.apply(lambda x: (x['a0'] - x['p0'])**2, axis=1)
    x = inflation_deltas.CPIAUCSL[47 + months:-1].tolist()
    prev = x[0]
    p1 = []
    for i in pmtx.p0:
        prev = prev * (i / 100 + 1)
        p1.append(prev)
    pmtx['p1'] = p1

    plots0 = {
        'inflation': pmtx.a0,
        'prediction': pmtx.p0
    }
    dfp0 = pd.DataFrame(plots0)

    plots1 = {
        'inflation': x,
        'prediction': pmtx.p1
    }
    dfp1 = pd.DataFrame(plots1)
    mse = pmtx['mse']

    return dfp0, dfp1, mse

    # fig = px.line(dfp1, y=dfp1.columns, x=dates[:])
    # fig2 = px.line(dfp0, y=dfp0.columns, x=dates[:])
    # fig2.show()
    # fig.show()
