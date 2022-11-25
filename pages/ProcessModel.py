# Import necessary libraries 
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from app import app
from functions import process_discovery, conformance_checking
import base64
import dash
import pickle, codecs

# discovered model store
model_store = dcc.Store(id='discovered-model', storage_type='local')

# conformance measure dropdown
conformance_dropdown = dcc.Dropdown(id='conformance-measure', options=['Fitness', 'Precision'], multi=False, value='Fitness')

# Define the page layout
layout = dbc.Container([
    dbc.Col([
        html.Center(html.H1("Process Discovery Original OCEL")),
        html.Hr(),
        model_store,
        dbc.Row([
            dbc.Col([html.Div("Select Conformance Measure:")], align='center'),
            dbc.Col([conformance_dropdown], align='center'),
            dbc.Col([dbc.Button("Calculate", id="calc-conformance", className="me-2", n_clicks=0)], align='center'),
            dbc.Col([html.Div("Result:")], align='center'),
            dbc.Col([html.Div(id='conformance-result')], align='center'),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Button("Show Process Model", color="warning", className="me-1", id='start-pm', n_clicks=0),
        ]),
        html.Div(id='pm-model'),
        html.Div(id='pm-model-display')
    ])
])

@app.callback(Output("conformance-result", "children"), [State("ocel_obj", "data"), State("conformance-measure", "value")], [Input("calc-conformance", "n_clicks")])
def on_button_click(ocel_obj, conformance_meas, n):
    if n > 0:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_obj.encode(), "base64"))
        # discover petri net
        ocpn = process_discovery.process_discovery_ocel_to_ocpn(ocel_log)
        # calculate fitness or precision, return as html div
        if conformance_meas == 'Fitness':
            fitness = conformance_checking.calculate_fitness(ocel_log, ocpn)
            result = html.Div(str(fitness))
        elif conformance_meas == 'Precision':
            precision = conformance_checking.calculate_precision(ocel_log, ocpn)
            result = html.Div(str(precision))
        return result
    else:
        return dash.no_update


@app.callback(Output("pm-model", "children"), State("ocel_obj", "data"), [Input("start-pm", "n_clicks")])
def on_button_click(ocel_obj, n):
    if n > 0:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_obj.encode(), "base64"))
        # discover and save petri net
        process_discovery.process_discovery_ocel_to_img(ocel_log, "oc_petri_net")
        return "Process Model successfully discovered."
    else:
        return dash.no_update


@app.callback(Output("discovered-model", "data"), State("discovered-model", "data"), Input("start-pm", "n_clicks"))
def on_click(data, n_clicks):
    # count clicks
    if n_clicks==0:
        raise PreventUpdate
    data = data or {'clicks': 0}
    data['clicks'] = data['clicks'] + 1
    return data
    

@app.callback(Output("pm-model-display", "children"), [Input("discovered-model", "data")], prevent_initial_call=True)
def on_button_click(data):
    data = data or {}
    clicks = data.get('clicks', 0)
    # if button is clicked, convert png to html image
    if clicks > 0:
        test_base64 = base64.b64encode(open('imgs/oc_petri_net.png', 'rb').read()).decode('ascii')
        return html.Img(src='data:image/png;base64,{}'.format(test_base64), style={'height':'70%', 'width':'70%'})
