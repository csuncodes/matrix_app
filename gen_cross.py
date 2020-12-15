import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import datetime
import io
import dash_table
import json
import dash_uploader as du

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# make a sample data frame with 6 columns
np.random.seed(0)
df = pd.DataFrame({"Col " + str(i+1): np.random.rand(30) for i in range(6)})
#print(df)
app.layout = html.Div([
    #dcc.Store(id='memory-output'),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '30%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div(id='output-data-upload'),
    html.Div(
        dcc.Graph(id='g1', config={'displayModeBar': False}),
        className='four columns'
        ),

    html.Div(
        dcc.Graph(id='g2', config={'displayModeBar': False}),
        className='four columns'
        ),
    html.Div(
        dcc.Graph(id='g3', config={'displayModeBar': False}),
        className='four columns'
    )
    ], className='row')

@app.callback(Output('intermediate-value', 'children'), [Input('upload-data', 'children')])
def clean_data(filename):
     # some expensive clean data step
    #content_type, content_string = contents.split(',')
    df = None
    #decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df = df.dropna()
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df = df.dropna()
    except Exception as e:
        print(e)
        df = None

     # more generally, this line would be
     # json.dumps(cleaned_df)
    print("HERE")
    print("IS")

    return df

def parse_contents(contents, filename, date):
    # Here is for file upload and parsing the data into df

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df = df.dropna()
            print(df)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df = df.dropna()
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'display': 'none'}
        )



@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        print(children)
        return children


def get_figure(df):
    # for an explanation
    fig = px.density_heatmap(df)

    fig.update_layout(width=500,
    height=500,margin=dict(l=20, r=0, t=15, b=5))

    return fig

def get_figure1(df, x_col, y_col, selectedpoints, selectedpoints_local):
    print("get_figure1")
    print(x_col)
    print(y_col)
    print(selectedpoints)
    print(selectedpoints_local)

    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation
    if selectedpoints_local != None:
        fig = px.histogram(selectedpoints, x=df[x_col], y=df[y_col])
    else:
        fig = px.histogram(selectedpoints)

    fig.update_traces(selectedpoints=selectedpoints,
                      customdata=df.index)



    return fig

def get_figure2(df, x_col, y_col, selectedpoints, selectedpoints_local):
    print(selectedpoints_local)

    if selectedpoints_local and selectedpoints_local['range']:
        ranges = selectedpoints_local['range']
        print("RANGES") #  goes here for selection
        print(ranges['x'][0])
        print(ranges['x'][1])
        print(ranges['y'][0])
        print(ranges['y'][1])
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}
    else:
        print("NPMIN") # goes here first
        print(np.min(df))
        print(np.max(df))
        print(np.min(df))
        print(np.max(df))
        selection_bounds = {'x0': np.min(df), 'x1': np.max(df),
                            'y0': np.min(df), 'y1': np.max(df)}

    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation
    if selectedpoints_local != None:
        fig = px.scatter(df, x=df[x_col], y=df[y_col])
    else:
        fig = px.scatter(df)

    fig.update_traces(selectedpoints=selectedpoints,
                      customdata=df.index)

    fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)

    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }},
                       **selection_bounds))

#--
    return fig
# this callback defines 3 figures
# as a function of the intersection of their 3 selections
@app.callback(
    [Output('g1', 'figure'),
    Output('g2', 'figure'),
     Output('g3', 'figure')],
    [Input('g1', 'selectedData'),
    Input('g2', 'selectedData'),
     Input('g3', 'selectedData'),
     Input('output-data-upload','children')]
)
def callback(selection1, selection2,selection3,value):
    print("callback")
    print(value)

    if isinstance(value, list):
        dic_trimmed = value[0]['props']['children'][2]['props']['data']
        y = json.dumps(dic_trimmed)
        value = pd.read_json(y)
        print(value)



    selectedpoints = df.index

    # need to have a df or value (value for changing selection points)
    for selected_data in [selection1, selection2,selection3]:
        print("1")
        print(selection1)
        print(selection2)
        print(selection3)
        
        if selected_data and selected_data['points']:
            print('2')
            print("ARGH")
            print(selected_data['points'])
            print(selected_data)
            selectedpoints = np.intersect1d(selectedpoints,
                [p['customdata'] for p in selected_data['points']])
            print('3')
            #need if statement for p, not all p has customdata
    print("FUCK")
    print(selectedpoints)
    return [get_figure(df),
            get_figure1(df, "Col 1", "Col 6", selectedpoints, selection1),
            get_figure2(df, "Col 1", "Col 6", selectedpoints, selection2)]


if __name__ == '__main__':
    app.run_server(debug=True)
