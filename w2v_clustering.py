import pickle
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

with open('../Databases/Processed/NYT_agg_processed_texts.pickle', 'rb') as f:
    agg_df = pickle.load(f)

# Define the start and end date for the rolling window
start_date = agg_df['date'].min()
end_date = datetime(year=2024, month=6, day=30) - relativedelta(years=4) # To ensure at least 4 years of data

dates = pd.date_range(start_date, end_date, freq='ME')

for current_date in dates:

    with open(f"/Users/rbomeny/Documents/Research - Marcelo/Databases/Data/Matrices/Clusters/Rolling/13clusters_{current_date.strftime('%Y%m')}_to_{(current_date + relativedelta(years=4)).strftime('%Y%m')}.pickle", 'rb') as handle:
        matrix1 = pickle.load(handle)
        matrix1['aux'] = 1
        matrix1 = matrix1[['date', 'cluster', 'aux']]

    cols = [i for i in range(0, 20)]

    df = matrix1.pivot(index='date',columns='cluster', values='aux')
    df.fillna(0, inplace=True)
    matrix1.index = matrix1['date']
    matrix1 = matrix1.join(df, how='outer')
    matrix1.drop(['aux', 'cluster'], axis=1, inplace=True)
    matrix1 = matrix1.groupby(pd.Grouper(freq='ME', key='date')).mean()
    # matrix1 = matrix1.apply(lambda x: sum(x[[i for i in range(0,20)]]), axis=1)
    with open(f"../Databases/Data/Matrices/Rolling_Clusters/matrix_ratios_{current_date.strftime('%Y%m')}_to_{(current_date + relativedelta(years=4)).strftime('%Y%m')}.pickle", 'wb') as f:
        pickle.dump(matrix1, f)
