import plotly.express as px
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)
with open('../Databases/Data/Matrices/Linear/l_predict_matrix_5k.pickle', 'rb') as f:
    pmtx1 = pickle.load(f)
with open('../Databases/Data/Matrices/Linear/cluster_treg.pickle', 'rb') as f:
    pmtx2 = pickle.load(f)

months = 1
month_dif = 1-months
if month_dif == 0:
    lmtx = [pmtx1[:], pmtx2[:]]
else:
    lmtx = [pmtx1[:month_dif], pmtx2[:month_dif]]

dates = inflation_deltas.index.tolist()[48+months:]



for pmtx in lmtx:
    pmtx.loc[:, 'p0'] = pmtx.predict.map(lambda x: x[months-1])
    pmtx.loc[:, 'a0'] = pmtx.actual.map(lambda x: x[months-1])
    pmtx['mse'] = pmtx.apply(lambda x: (x['a0']-x['p0'])**2, axis=1)
    x = inflation_deltas.CPIAUCSL[48+months:].tolist()[0]
    prev = x
    p1 = []
    for i in pmtx.p0:
        prev = prev*(i/100+1)
        p1.append(prev)
    pmtx['p1'] = p1

plots0 = {
    'inflation':lmtx[0].a0,
    '25k words':lmtx[0].p0,
    'cluster':lmtx[1].p0
}
dfp0 = pd.DataFrame(plots0)

plots1 = {
    'inflation':inflation_deltas['CPIAUCSL'][48+months:].tolist(),
    '25k words':lmtx[0].p1,
    'cluster':lmtx[1].p1
}
dfp1 = pd.DataFrame(plots1)

fig = px.line(dfp1, y=dfp1.columns, x=dates[:])
fig2= px.line(dfp0, y=dfp0.columns, x=dates[:])
fig2.show()
fig.show()