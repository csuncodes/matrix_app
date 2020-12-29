import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from chord import Chord
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
import dash_uploader as du
import uuid
from sklearn.decomposition import PCA

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
UPLOAD_FOLDER_ROOT = r"C:\tmp\Uploads"
du.configure_upload(app, UPLOAD_FOLDER_ROOT)
# make a sample data frame with 6 columns
np.random.seed(0)

df_iris = datasets.load_iris()
df = df_iris.data
df = pd.DataFrame(df)

#pca = PCA()
#pca.fit(df)
#exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
#x=range(1, exp_var_cumul.shape[0] + 1)
app.layout = html.Div(
    [

        # empty Div to trigger javascript file for graph resizing

        html.Div(
            [
                

                html.Div(
                    [
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="rajanlab.com",
                        )
                    ],
                    className="one-third column",
                    id="button",
                    style={
                        "height": "60px",
                        "width": "auto",
                        "margin-bottom": "25px",
                    },
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(style={'backgroundColor': "#d0d1ff",'margin':15}, children=
                    [
                        html.P(
                            "Try your own data!",
                            className="control_label",
                        ),

                                        du.Upload(
                                            id='dash-uploader',
                                            max_file_size=1800,  # 1800 Mb
                                            filetypes=['csv', 'zip'],
                                            upload_id=uuid.uuid1(),  # Unique session id
                                        ),
                                        html.Div(id='callback-output'),


                        html.Div(id='intermediate-value', style={'display': 'none'}),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options"

                ),

         html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id='scatter', config={'displayModeBar': False})],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                    style={'margin':5}
                    ),
            ],
        ),
        html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id='g3', config={'displayModeBar': False})],
                            className="pretty_container",
                        ),
                    ],
                    className="twelve columns",
                    style={'margin':5}
                    ),
        html.Div(
            [
                html.Div(
                    [ dcc.Graph(id='g2', config={'displayModeBar': False})],
                    className="pretty_container seven columns",
                    style={'margin':5}
                ),
                html.Div(
                    [dcc.Graph(id='g5', config={'displayModeBar': False})],
                    className="pretty_container five columns",
                    style={'margin':5}
                ),
            ],
            className="row flex-display",
            id='graph-output'
        ),
         html.Div(
            [
                html.Div(
                    [dcc.Graph(id='scatter_matrix', config={'displayModeBar': False}),dcc.Input(id="n_comp1", type="number", placeholder="# of Components",min=0)],
                    className="pretty_container seven columns",
                    style={'margin':5}
                ),
                html.Div(
                    [dcc.Graph(id='g6', config={'displayModeBar': False}),dcc.Input(id="input1", type="number", placeholder="X Variable",min=0),
                    dcc.Input(id="input2", type="number", placeholder="Y Variable",min=0)],
                    className="pretty_container five columns",
                    style={'margin':5}
                ),

            ],
            className="row flex-display",
        )

    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column",'backgroundColor': "#d0d1ff"},
)


#app.layout = html.Div([

#        html.Div(
#            [
#                du.Upload(
#                    id='dash-uploader',
#                    max_file_size=1800,  # 1800 Mb
#                    filetypes=['csv', 'zip'],
#                    upload_id=uuid.uuid1(),  # Unique session id
#                ),
#                html.Div(id='callback-output'),
#            ],
#            style={  # wrapper div style
#                'textAlign': 'center',
#                'width': '600px',
#                'padding': '10px',
#                'display': 'inline-block'
#            }),

#    dcc.Dropdown(
#        id='crossfilter-yaxis-column',
#        options=[{'label':'None','value':'None'},
#        {'label':'PCA','value':'PCA'},
#        {'label':'SVD','value':'SVD'},
#        {'label':'Isomap','value':'ISO'},
#        {'label':'Local Linear Embeddings', 'value':'LLE'},
#        {'label':'t-SNE','value':'TSNE'}],
#        value='None',
#        style=dict(
#                    width='40%'
#                )
#    ),
#    dcc.Input(id="input1", type="text", placeholder="X Variable"),
#    dcc.Input(id="input2", type="text", placeholder="Y Variable"),
#    html.Div(id='intermediate-value', style={'display': 'none'}),
#    html.Div(
#        dcc.Graph(id='scatter', config={'displayModeBar': True}),
#        className='four columns'
#    ),

#    html.Div(
#        dcc.Graph(id='g2', config={'displayModeBar': True},style=dict(
#                    width='150%',height='200%'
#                )),
#        className='four columns'
#        ),
#    html.Div(
#        dcc.Graph(id='g3', config={'displayModeBar': True}),
#        className='four columns'
#    ),
#    html.Div(
#        dcc.Graph(id='g4', config={'displayModeBar':True},style=dict(
#                    width='150%',height='200%'
#                )),
#        className='four columns'
#    )
#], className='row')


def get_figure( projections):
    print('SCATTER')

    #print(selectedpoints_local)

    #if selectedpoints_local and selectedpoints_local['range']:
    #    ranges = selectedpoints_local['range']

    #    selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
    #                        'y0': ranges['y'][0], 'y1': ranges['y'][1]}
    #else:

    #    selection_bounds = {'x0': np.min(df), 'x1': np.max(df),
    #                        'y0': np.min(df), 'y1': np.max(df)}

    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation

    fig = px.scatter(projections,color_continuous_scale='sunsetdark')



    fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)



    return fig

def get_figure2(explained_var):
    print("COMP")



    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21

    exp_var_cumul = np.cumsum(explained_var)

    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
        )


    #fig = px.histogram(selectedpoints)
    #fig.update_layout(autosize=True,margin=dict(l=30, r=30, b=20, t=40),hovermode="closest",plot_bgcolor="#F9F9F9",paper_bgcolor="#E9E9E9",legend=dict(font=dict(size=10), orientation="h"))
    #fig = px.scatter(df, x=df[x_col], y=df[y_col], text=df.index)
    #print("HISTO POINTS")
    #print(selectedpoints)

    #fig.update_layout(margin={'l': 20, 'r': 0, 'b': 15, 't': 5}, dragmode='select', hovermode=False)

    #fig.add_shape(dict({'type': 'rect',
                        #'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }},
                       #**selection_bounds))
    return fig

def get_figure3(df):
    print('MATRIX')
    print(df)
    #df = df.loc[:, :'sepal_width']
    #df = df.corr('pearson')
    #print(df)
    #px.density_heatmap(selectedpoints)
    #a = np.expand_dims(selectedpoints, axis=0)
    fig = px.imshow(df)
    fig.update_layout(autosize=True,margin=dict(l=30, r=30, b=20, t=40),hovermode="closest",plot_bgcolor="#F9F9F9",paper_bgcolor="#E9E9E9",legend=dict(font=dict(size=10), orientation="h"))

    return fig


def get_figure4(n_components, components,total_var):
    print('3D')


    fig = px.scatter_3d(
        components, x=0, y=1, z=2,color=0,color_continuous_scale='sunsetdark',
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
    fig.update_layout(autosize=True,margin=dict(l=30, r=30, b=20, t=40),hovermode="closest",plot_bgcolor="#F9F9F9",paper_bgcolor="#E9E9E9",legend=dict(font=dict(size=10), orientation="h"))
    return fig

def get_figure5(df, x_col, y_col):
    print("HEATMAP")
    print(df)
    fig = px.density_heatmap(df, x=x_col, y=y_col, marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(autosize=True,margin=dict(l=30, r=30, b=20, t=40),hovermode="closest",plot_bgcolor="#F9F9F9",paper_bgcolor="#E9E9E9",legend=dict(font=dict(size=10), orientation="h"))
    return fig




def get_figure6(n_components,components,total_var):
    print("SCATTER MATRIX")

    labels = {str(i): f"PC {i+1}"
              for i in range(n_components)}
    labels['color'] = 'Median Price'

    fig = px.scatter_matrix(
        components,
        color=0,
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}%')
    fig.update_traces(diagonal_visible=False)

    return fig

@du.callback(
    output=Output('callback-output', 'children'),
    id='dash-uploader',
)
def get_a_list(filenames):
    print(filenames)
    print(pd.read_csv(filenames[0]))
    return filenames



# this callback defines 3 figures
# as a function of the intersection of their 3 selections
@app.callback(
    [Output('scatter', 'figure'),
    Output('g5', 'figure'),
    Output('g3', 'figure'),
    Output('g6', 'figure'),
    Output('scatter_matrix', 'figure'),
    Output("g2", "figure")],
    [
    Input("input1", 'value'),
     Input("input2", 'value'),
     Input('callback-output', 'children'),
     Input("n_comp1", 'value')
    ]
)
def callback(x_col,y_col,upload,n_comp):
    #features = df.loc[:, :'sepal_width']
    print("BJL")
    print(upload)
    if upload is not None:
        df = pd.read_csv(upload[0])
        df = df.dropna()
    else:

        df = df_iris.data
        df = pd.DataFrame(df)

    n_neighbors = 30


    projections = df

    selectedpoints = df.index
    pca = PCA(n_components=3)
    components = pca.fit_transform(df)
    total_var = pca.explained_variance_ratio_.sum() * 100
    explained_var = pca.explained_variance_ratio_
    #for selected_data in [selection1, selection2]:
    #    if selected_data and selected_data['points']:
    #        print(selected_data['points'])
    #        if 'pointNumbers' in selected_data['points'][0]:

    #            for p in selected_data['points']:
    #                selectedpoints = np.intersect1d(selectedpoints,[p['pointNumbers']])
    #        else:
    #            selectedpoints = np.intersect1d(selectedpoints,
    #               [p['pointNumber'] for p in selected_data['points']])

    #for selected_data in [selection1, selection2,selection3, selection4]:
    #    print("1")
    #    if selected_data and selected_data['points']:
    #        print('2')
            #print(selected_data['points'])

    #        selectedpoints = np.intersect1d(selectedpoints,
    #            [p['customdata'] for p in selected_data['points']])
    #        print('3')


    if x_col == None:
        x_col = df.columns[0]
    if y_col == None:
        y_col = df.columns[1]


    return [get_figure(components),
            get_figure2(explained_var),
            get_figure3(df),
            get_figure5(df, x_col, y_col),
            get_figure6(3,components,total_var),
            get_figure4(3, components,total_var)]


if __name__ == '__main__':
    app.run_server(debug=True)
