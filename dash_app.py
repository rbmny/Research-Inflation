import pickle
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import os

with open('../Databases/Data/Inflation/inflation_deltas.pkl', 'rb') as f:
    inflation_deltas = pickle.load(f)
dates = inflation_deltas.index.tolist()[48:-1]

main_folders_path = "../Databases/Data/Matrices/"
type_folders = os.listdir(main_folders_path)
type_folders = [f for f in type_folders if os.path.isdir(os.path.join(main_folders_path, f))]
type_folders.remove('Rolled_7')
type_folders.remove('Rolled_30')

# path = "../Databases/Data/Matrices/Elastic/Coefs"
# dir_list = os.listdir(path)
# dir_list = [f for f in dir_list if os.path.isfile(path + '/' + f)]
# dir_list.remove(".DS_Store")


from plots_called import plot

app = Dash()

app.layout = html.Div([
    html.H1(children='Dashboard Previsões de Inflação', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label('Choose Folder'),
            dcc.Dropdown(type_folders, id='dropdown-selection-m-folder',value='Elastic', style={'color': 'black', 'width': 300})]),

        html.Div([
            html.Label('Choose Sub-folder (optional)'),
            dcc.Dropdown(id='dropdown-selection-s-folder', style={'color': 'black', 'width': 300})]),

        html.Div([
            html.Label('Choose File'),
            dcc.Dropdown(id='dropdown-selection', style={'color': 'black', 'width': 300})]),
        html.Div([
            dcc.Input(id="input-number", type='number', placeholder="Months ahead", value=1, style={'color': 'black', 'width': 300})])
    ], style={'padding': 10, 'flex': 1, 'display': 'flex', 'flex-direction': 'row', 'align-items': 'center',
              'justify-content': 'center', 'gap': 10}),

    html.Br(),
    dcc.Graph(id='graph-content'),
    dcc.Graph(id='graph-content-2'),
    dcc.Graph(id='graph-content-3')
])


@callback(
    Output('dropdown-selection-s-folder', 'options'),
    Input('dropdown-selection-m-folder', 'value'),
)
def update_subfolder_options(selected_folder):
    path = os.path.join(main_folders_path, selected_folder)
    dir_list = os.listdir(path)
    dir_list = [f for f in dir_list if os.path.isdir(path + '/' + f)]
    sub_list = [i+'/'+f for i in  dir_list for f in os.listdir(f'{path}/{i}') if os.path.isdir(f'{path}/{i}/{f}')]
    dir_list += sub_list
    try:
        dir_list.remove('.DS_Store')
    except:
        print('no DS.store')
    return dir_list


@callback(
    Output('dropdown-selection', 'options'),
    Input('dropdown-selection-m-folder', 'value'),
    Input('dropdown-selection-s-folder', 'value'),
)
def update_file_options(selected_folder, sub_folder):
    if sub_folder is not None:
        path = os.path.join(main_folders_path, selected_folder, sub_folder)
    else:
        path = os.path.join(main_folders_path, selected_folder)
    dir_list = os.listdir(path)
    dir_list = [f for f in dir_list if os.path.isfile(path + '/' + f)]
    try:
        dir_list.remove('.DS_Store')
    except:
        print('no DS.store')
    return dir_list


@callback(
    Output('graph-content', 'figure'),
    Output('graph-content-2', 'figure'),
    Output('graph-content-3', 'figure'),

    Input('input-number', 'value'),
    Input('dropdown-selection', 'value'),
    Input('dropdown-selection-m-folder', 'value'),
    Input('dropdown-selection-s-folder', 'value')
)
def update_graph(months, file_name, main_folder, sub_folder):
    if sub_folder is not None:
        path = os.path.join(main_folders_path, main_folder, sub_folder, file_name)
    else:
        path = os.path.join(main_folders_path, main_folder, file_name)
    df = pd.read_pickle(path)
    dfp0, dfp1, mse = plot(df, months)
    return (px.line(dfp1, y=['inflation', 'prediction'], x=dates[months-1:]),
            px.line(dfp0, y=dfp0.columns, x=dates[months-1:]),
            px.line( y=mse, x=dates[months-1:]))


if __name__ == '__main__':
    app.run(debug=False)
