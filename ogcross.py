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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
UPLOAD_FOLDER_ROOT = r"C:\tmp\Uploads"
du.configure_upload(app, UPLOAD_FOLDER_ROOT)
# make a sample data frame with 6 columns
np.random.seed(0)
df_iris = datasets.load_iris()


app.layout = html.Div(
    [

        # empty Div to trigger javascript file for graph resizing

        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "New York Oil and Gas",
                                    style={"margin-bottom": "5px"},
                                ),
                                html.H5(
                                    "Production Overview", style={"margin-top": "5px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="https://plot.ly/dash/pricing/",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(style={'backgroundColor': "#E9E9E9",'margin':15}, children=
                    [
                        html.P(
                            "Filter by construction date (or select range in histogram):",
                            className="control_label",
                        ),

                                        du.Upload(
                                            id='dash-uploader',
                                            max_file_size=1800,  # 1800 Mb
                                            filetypes=['csv', 'zip'],
                                            upload_id=uuid.uuid1(),  # Unique session id
                                        ),
                                        html.Div(id='callback-output'),

                         dcc.Dropdown(
                                id='crossfilter-yaxis-column',
                                options=[{'label':'None','value':'None'},
                                {'label':'PCA','value':'PCA'},
                                {'label':'SVD','value':'SVD'},
                                {'label':'Isomap','value':'ISO'},
                                {'label':'Local Linear Embeddings', 'value':'LLE'},
                                {'label':'t-SNE','value':'TSNE'}],
                                value='None'
                        ),
                        dcc.Input(id="input1", type="text", placeholder="X Variable"),
                        dcc.Input(id="input2", type="text", placeholder="Y Variable"),
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
                    className="seven columns",
                    style={'margin':5}
                    ),
            ],
        ),

        html.Div(
            [
                html.Div(
                    [ dcc.Graph(id='g2', config={'displayModeBar': False})],
                    className="pretty_container seven columns",
                    style={'margin':5}
                ),
                html.Div(
                    [dcc.Graph(id='g3', config={'displayModeBar': False})],
                    className="pretty_container five columns",
                    style={'margin':5}
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id='g4', config={'displayModeBar': False})],
                    className="pretty_container seven columns",
                    style={'margin':5}
                )

            ],
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column",'backgroundColor': "#ced4da"},
)


#app.layout = html.Div([
#    dcc.ConfirmDialog(
#        id='confirm',
#        message='Please input a valid X or Y variable column',
#    ),
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

#@app.callback(Output('confirm', 'displayed'),
#              Input('dropdown', 'value'))
#def display_confirm(value):
#    if value == 'Danger!!':
#        return True
#    return False

def get_figure(df, projections, x_col, y_col, selectedpoints, selectedpoints_local):

    if selectedpoints_local and selectedpoints_local['range']:
        ranges = selectedpoints_local['range']
        selection_bounds = {'x0': ranges['x'][0], 'x1': ranges['x'][1],
                            'y0': ranges['y'][0], 'y1': ranges['y'][1]}

    else:
        selection_bounds = {'x0': np.min(df[x_col]), 'x1': np.max(df[x_col]),
                            'y0': np.min(df[y_col]), 'y1': np.max(df[y_col])}
     #somethings wrong here

    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation


    fig = px.scatter(
        df
    )
    #x=0,y=1
    #color=df.species, labels={'color': 'species'}
    #fig = px.scatter(df, x=df[x_col], y=df[y_col], text=df.index)
    #print("SCATTER POINTS")
    #print(selectedpoints)
    fig.update_traces(selectedpoints=selectedpoints,
                      customdata=df.index,
                      mode='markers+text', marker={'size': 5}, unselected={'marker': { 'opacity': 0.3 }, 'textfont': { 'color': 'rgba(0, 0, 0, 0)' }})
    #print("UPDATE SCATTER TRACES")
    fig.update_layout(autosize=True,margin=dict(l=30, r=30, b=20, t=40),hovermode="closest",plot_bgcolor="#F9F9F9",paper_bgcolor="#E9E9E9",legend=dict(font=dict(size=10), orientation="h"))
    fig.update_yaxes(automargin=True)
    fig.add_shape(dict({'type': 'rect',
                        'line': { 'width': 1, 'dash': 'dot', 'color': 'darkgrey' }},
                       **selection_bounds))
    return fig

def get_figure2(df, projections, x_col, y_col, selectedpoints, selectedpoints_local):



    # set which points are selected with the `selectedpoints` property
    # and style those points with the `selected` and `unselected`
    # attribute. see
    # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
    # for an explanation
    print("HERE")
    print(selectedpoints)
    print(selectedpoints_local)
    if selectedpoints_local != None:
        fig = px.histogram(selectedpoints, x=df[x_col], y=df[y_col])
    else:
        fig = px.histogram(selectedpoints)

    fig.update_traces(selectedpoints=selectedpoints,
                      customdata=df.index)



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

def get_figure3(df, projections, x_col, y_col, selectedpoints, selectedpoints_local):
    #df = df.loc[:, :'sepal_width']
    #df = df.corr('pearson')
    #print(df)
    #px.density_heatmap(selectedpoints)
    a = np.expand_dims(selectedpoints, axis=0)
    fig = px.imshow(a)
    fig.update_layout(autosize=True,margin=dict(l=30, r=30, b=20, t=40),hovermode="closest",plot_bgcolor="#F9F9F9",paper_bgcolor="#E9E9E9",legend=dict(font=dict(size=10), orientation="h"))

    return fig


def get_figure4(df, projections, x_col, y_col, selectedpoints, selectedpoints_local):
    #df = px.data.iris()
    #X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    X = df.columns
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter_3d(
        components, x=0, y=1, z=2,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
    fig.update_layout(autosize=True,margin=dict(l=30, r=30, b=20, t=40),hovermode="closest",plot_bgcolor="#F9F9F9",paper_bgcolor="#E9E9E9",legend=dict(font=dict(size=10), orientation="h"))
    return fig

def get_figure5(df, projections, x_col, y_col, selectedpoints, selectedpoints_local):
    fig = px.density_heatmap(df, x=x_col, y=y_col, marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(autosize=True,margin=dict(l=30, r=30, b=20, t=40),hovermode="closest",plot_bgcolor="#F9F9F9",paper_bgcolor="#E9E9E9",legend=dict(font=dict(size=10), orientation="h"))
    return fig

@du.callback(
    output=Output('callback-output', 'children'),
    id='dash-uploader',
)
def get_a_list(filenames):
    print(filenames)
    print(pd.read_csv(filenames[0]))
    return html.Ul([html.Li(filenames)])
# this callback defines 3 figures
# as a function of the intersection of their 3 selections
@app.callback(
    [Output('scatter', 'figure'),
     Output('g2', 'figure'),
     Output('g3', 'figure'),
     Output('g4', 'figure')],
    [Input('crossfilter-yaxis-column', 'value'),
    Input('scatter', 'selectedData'),
     Input('g2', 'selectedData'),
     Input('g3', 'selectedData'),
     Input('g4', 'selectedData'),
     Input("input1", 'value'),
     Input("input2", 'value')]
)
def callback(analysis,selection1, selection2, selection3, selection4,x_col,y_col):
    #features = df.loc[:, :'sepal_width']

    df = df_iris.data[:, :4]
    df = pd.DataFrame(df)
    print(len(df.columns))
    n_neighbors = 30
    print(analysis)
    #if analysis == 'SVD' or analysis == 'PCA':
    #    print("a")
    #    projections = decomposition.TruncatedSVD(n_components=1).fit_transform(df)
    #elif analysis == 'ISO':
    #    print("b")
    #    projections = manifold.Isomap(n_neighbors=n_neighbors, n_components=1
    #                    ).fit_transform(df)
    #elif analysis == 'LLE':
    #    print("c")
    #    clf = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=1,
    #                                  method='standard')
    #    projections = clf.fit_transform(df)
    #elif analysis == 'TSNE':
    #    print('d')
    #    tsne = TSNE(n_components=2, random_state=0)
    #    projections = tsne.fit_transform(df)
    #else:
    projections = df

    selectedpoints = df.index

    #for selected_data in [selection1, selection2, selection3, selection4]:
        #if selected_data and selected_data['points']:
        #    if 'pointNumbers' in selected_data['points'][0]:

        #        for p in selected_data['points']:
        #            selectedpoints = np.intersect1d(selectedpoints,[p['pointNumbers']])
        #    else:
        #        selectedpoints = np.intersect1d(selectedpoints,
        #            [p['customdata'] for p in selected_data['points']])

    for selected_data in [selection1, selection2,selection3, selection4]:
        print("1")
        print(selection1)
        print(selection2)
        print(selection3)
        print(selection4)
        if selected_data and selected_data['points']:
            print('2')
            #print(selected_data['points'])

            selectedpoints = np.intersect1d(selectedpoints,
                [p['customdata'] for p in selected_data['points']])
            print('3')





    print("FUCK")
    print(selectedpoints)
    if x_col == None:
        x_col = df.columns[0]
    if y_col == None:
        y_col = df.columns[1]
    #if x_col not in df.columns:



    return [get_figure(df, projections, x_col, y_col, selectedpoints, selection2),
            get_figure2(df, projections, x_col, y_col, selectedpoints, selection1),
            get_figure3(df, projections, x_col, y_col, selectedpoints, selection1),
            get_figure5(df, projections, x_col, y_col, selectedpoints, selection1)]


if __name__ == '__main__':
    app.run_server(debug=True)
