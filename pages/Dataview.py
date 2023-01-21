# Import necessary libraries 
from dash import html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
from app import app
from dash.dependencies import Input, Output
from functions import dataimport
import pickle, codecs

# create empty Dataframe that is displayed before uploading any logs
dummy_df = pd.DataFrame(columns=['event_id', 'activity', 'timestamp', 'object'])

# create DataTable
ocel_table = dbc.Table.from_dataframe(dummy_df, striped=True, bordered=True, hover=True, id="ocel_table_full")

# Define the page layout
layout = dbc.Container([
    dbc.Row([
        html.Center(html.H1("View OCEL")),
        html.Br(),
        html.Hr(),
        html.H4("Summary of OCEL"),
        html.Hr(),
    ]),
    dbc.Row([
        html.Div(id="ocel-summary-text")
        ]),
    dbc.Row([
        html.H4("OCEL (limited to 1000 records)"),
        html.Hr(),
    ]),
    dbc.Row([
        html.Div([ocel_table], style={"overflow": "scroll"})
    ])
])

@app.callback(Output("ocel_table_full", "children"), Input("ocel_obj", "data"))
def on_upload_ocel_full(ocel_log):
    if ocel_log is None:
        return dummy_df
    else:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
        # convert ocel object to dataframe and return as DataTable
        ocel_df, _ = dataimport.ocel_to_df_params(ocel_log)
        # cut ocel if too large
        if len(ocel_df) > 1000:
            ocel_df = ocel_df.head(1000)
        return dbc.Table.from_dataframe(ocel_df, striped=True, bordered=True, hover=True)


@app.callback(Output("ocel-summary-text", "children"), [Input("param-store", "data")])
def on_upload_params(params):
    if params != None:
        # create summary text out of parameter store
        ocel_summary_text = html.P([
            html.H6("Object Types: "), str(params['object_types']), html.H6("Activitiy Count: "), str(params['activity_count']), html.H6("Object Type Count: "), str(params['object_type_occurences']),
            html.H6("Number of Events: "), str(params['num_events']), html.H6("Number of Activities: "), str(params['num_activities']), html.H6("Number of Objects: "), str(params['num_objects']),
            ])
        return ocel_summary_text
    else:
        return []