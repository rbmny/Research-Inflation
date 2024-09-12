import plotly_express as px
import pickle
import pandas as pd
from dateutil.relativedelta import relativedelta

with open('../Databases/Data/Matrices/Linear/cluster_treg.pickle', 'rb') as f:
    cluster = pickle.load(f)
    cluster.loc[:, 'p0'] = cluster.predict.map(lambda x: x[0])
with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)

x = inflation_deltas.CPIAUCSL[47:-1].tolist()
prev = x[0]
p1 = []
for i in cluster.p0:
    prev = prev * (i / 100 + 1)
    p1.append(prev)
cluster['p1'] = p1

cluster['date'] = [i + relativedelta(years=4) for i in cluster.index.tolist()]
cluster.index = cluster.date.apply(pd.to_datetime)
fig1 = px.line(cluster, y=['p0', 'actual'])
fig2 = px.line(cluster, y=['p1', x[:-1]])
fig1.show()
fig2.show()
print()