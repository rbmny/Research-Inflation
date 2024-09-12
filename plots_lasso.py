import plotly.express as px
import pickle
import pandas as pd

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)
with open('../Databases/Data/Matrices/Predict/predict_related_words.pickle', 'rb') as f:
    pmtx1 = pickle.load(f)
with open('../Databases/Data/Matrices/Predict/predict_matrix_15k.pickle', 'rb') as f:
    pmtx2 = pickle.load(f)
lmtx = [pmtx1, pmtx2]

dates = inflation_deltas.index.tolist()[49:]

for pmtx in lmtx:
    pmtx.loc[:, 'p0'] = pmtx.predict1.map(lambda x: x[0])
    pmtx.loc[:, 'a0'] = pmtx.actual.map(lambda x: x)
    x = inflation_deltas.CPIAUCSL[49:].tolist()
    prev = x[0]
    p1 = []
    for i in pmtx.p0:
        prev = prev*(i/100+1)
        p1.append(prev)
    pmtx['p1'] = p1

plots0 = {
    'inflation':lmtx[0].a0,
    'trigram_common_10':lmtx[0].p0,
    '15k words':lmtx[1].p0
}
dfp0 = pd.DataFrame(plots0)

plots1 = {
    'inflation':inflation_deltas['CPIAUCSL'][49:].tolist(),
    'trigram_common_10':lmtx[0].p1,
    '15k words':lmtx[1].p1
}
dfp1 = pd.DataFrame(plots1)

fig = px.line(dfp1, y=dfp1.columns, x=dates[:])
fig2= px.line(dfp0, y=dfp0.columns, x=dates[:])
fig2.show()
fig.show()