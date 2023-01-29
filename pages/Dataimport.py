# Import necessary libraries 
from dash import html, dcc, ctx
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from app import app
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import Trigger, ServersideOutput
from dash.exceptions import PreventUpdate
import os
import pickle
import codecs

from functions import dataimport, process_executions
from components import explanation_texts, input_forms

import dash_uploader as du
# configure the Dash uploader component
UPLOAD_FOLDER_ROOT = "assets/"
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=True)
my_upload = du.Upload(id='dash-uploader', text='Drag & Drop here to upload a file', text_completed = 'File is now available (click REFRESH): ', filetypes=['jsonocel', 'xmlocel', 'csv'], upload_id="uploaded_logs")
# define Dropdown for available logs
my_uploaded_files = dcc.Dropdown(placeholder='Select an OCEL', id='uploaded-logs', options=[{'label':file, 'value':file} for file in os.listdir("assets/uploaded_logs")])

# create empty Dataframe to display before any OCEL is uploaded
dummy_df = pd.DataFrame(columns=['event_id', 'activity', 'timestamp', 'object'])
# create empty DataTable used as placeholder
ocel_table = dbc.Table.from_dataframe(dummy_df, striped=True, bordered=True, hover=True, id="ocel_table_head")

# create store for csv parameters
csv_params = dcc.Store(id='csv-params', storage_type='local')

# create Dropdown used for leading object type selection
leading_object_div = html.Div([ dcc.Dropdown(placeholder='Select leading object type', id='leading-object', style={'display': 'block', 'width':'80%'})])

# create radio items selection for leading object type oder connected component process execution extraction
process_extraction = html.Div([
                                dbc.Row([
                                    dbc.Col([html.Div("Select Type of Process Execution Extraction: "), dbc.RadioItems(options=[{"label": "Connected Components", "value": "CONN_COMP"},{"label": "Leading Object Type", "value": 'LEAD_TYPE '}], value="CONN_COMP", id="process-extraction-type"), leading_object_div]),
                                    dbc.Col([explanation_texts.extraction_type_explanation], width=8)
                                    ])
                                ])

# Define the page layout
layout = dbc.Container([
        csv_params,
        html.Center(html.H1("Import Data")),
        html.Hr(),
        my_upload,
        html.Hr(),
        html.H5("Select an OCEL for import:"),
        dbc.Row([dbc.Col([my_uploaded_files], width=8), dbc.Col([dbc.Button("Refresh", className="me-2", color='warning', id='refresh-list', n_clicks=0)])]),
        html.Br(),
        input_forms.csv_import,
        html.Div("Parameters successfully parsed.", style={'display':'none'}, id='success-parse-csv'),
        html.Hr(),
        process_extraction,
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Button("Upload", color="warning", id="upload-button", className="me-1", n_clicks=0, disabled=True),
                dbc.Button("Cancel", className="me-2", id='cancel-upload', n_clicks=0)
            ],width=7), 
        ]), 
        dbc.Row(html.Progress(id="progress-bar-upload", value="0")),
        html.Div("OCEL successfully uploaded.", style={'display':'none'}, id='success-upload-ocel'),
        html.Br(),
        html.Div([
            ocel_table
            ], id='ocel-table', style={"overflow": "scroll"})
    ])

'''
# update file dropdown on upload
@du.callback(
    output=Output("uploaded-logs", "options"),
    id="dash-uploader",
)
def callback_on_completion(status: du.UploadStatus):
    if status.is_completed == True: 
        updated_options = [{'label':file, 'value':file} for file in os.listdir("assets/uploaded_logs/")]
        return updated_options
    else:
        return dash.no_update
'''

# read files from uploaded_logs directory if refresh button is clicked
@app.callback(Output("uploaded-logs", "options"), Input('refresh-list', 'n_clicks'), prevent_initial_call=True)
def on_refresh_files(n):
    if n>0:
        updated_options = [{'label':file, 'value':file} for file in os.listdir("assets/uploaded_logs/")]
        return updated_options
    else:
        return dash.no_update

# load csv parameters into store, if parse button is clicked, uncover successful parsing div
@app.callback([Output("csv-params", "data"), Output("success-parse-csv", "style")], [State("obj_names", "value"), State("act_name", "value"), State("start_time_name", "value"), State("time_name", "value"), State("id_name", "value"),], Input("parse-csv",  "n_clicks"), prevent_initial_call=True)
def on_upload_csv(obj_name, act_name, start_time_name, time_name, id_name, n):
    # default seperator
    sep = ","
    # case distinction on existence of start timestamp
    if start_time_name != None: 
        params = {
            "obj_names":obj_name,
            "val_names":[],
            "act_name":act_name,
            "start_timestamp":start_time_name,
            "time_name":time_name,
            "id_name":id_name,
            "sep":sep
        }
    else:
        params = {"obj_names":obj_name,
                "val_names":[],
                "act_name":act_name,
                "time_name":time_name,
                "id_name":id_name,
                "sep":sep}
    return params, {'display':'block'}

# load possible leading object types into dropdown
@app.callback(Output("leading-object", "options"), [Input("csv-params","data"), Input("process-extraction-type", "value"), Input("uploaded-logs", "value")], prevent_initial_call=True)
def on_parse_params(csv_params, process_ex_type, uploaded_file):
    if uploaded_file != None:
        # read object types from parameter input form in case of OCEL in csv format
        if (csv_params != None) and (uploaded_file.endswith('csv')):
            options = csv_params['obj_names']
            return options 
        else:
            # if connected components is selected, leave list empty
            if process_ex_type == "CONN_COMP":
                return [] 
            else: 
                # process OCEL and extract object types
                ocel_log = dataimport.load_ocel_json_xml("assets/uploaded_logs/" + uploaded_file, parameters=None)
                object_types = dataimport.get_ocel_object_types(ocel_log)
                return object_types
    else:
        return []


# enable upload button if file is selected and parameters are parsed in case of OCEL in csv format
@app.callback(Output("upload-button", "disabled"), [Input("success-parse-csv", "style"), Input("uploaded-logs", "value")], prevent_initial_call = True)
def on_file_selection(csv_params_parsed, selected_file):
    if selected_file.endswith("csv") == False:
        return False 
    elif csv_params_parsed == {'display':'block'}:
        return False 
    else:
        raise PreventUpdate

# load and store ocel, extract and store parameters, uncover 'success' div
@app.long_callback(output=(
    ServersideOutput("ocel_obj", "data"), 
    Output("param-store", "data"), 
    ServersideOutput("execution-store", "data"), 
    Output("success-upload-ocel", "style")
    ),inputs=(
        State("uploaded-logs", "value"), 
        State("csv-params", "data"),
        State("process-extraction-type", "value"),
        State("leading-object", "value"), 
        Trigger("upload-button",  "n_clicks")), 
    running=[
        (Output("upload-button", "disabled"), True, False),
        (Output("cancel-upload", "disabled"), False, True),
        (Output("progress-bar-upload", "style"),{"visibility": "visible"},{"visibility": "hidden"}),
        ],
    cancel=[Input("cancel-upload", "n_clicks")],
    progress=[Output("progress-bar-upload", "value"), Output("progress-bar-upload", "max")],
    prevent_initial_call=True)
def on_upload_ocel_path(set_progress, uploaded_file, csv_params, process_extr_type, leading_obj, n):
    if n > 0:
        # start progress bar 
        set_progress(("0","10"))
        if uploaded_file is None:
            raise PreventUpdate
        else:
            # use different load function w.r.t file extension
            if uploaded_file.endswith("csv"):
                ocel_df = pd.read_csv("assets/uploaded_logs/" + uploaded_file)
                ocel_df = dataimport.remove_prefix_csv(ocel_df)
                if process_extr_type == "CONN_COMP":
                    ocel_log = dataimport.df_to_ocel(ocel_df, csv_params)
                else:
                    # add leading object for process execution extraction
                    csv_params["execution_extraction"] = "leading_type"
                    csv_params["leading_type"] = leading_obj
                    ocel_log = dataimport.df_to_ocel(ocel_df, csv_params)
            # case OCEL in xmlocel or jsonocel format
            else:
                if process_extr_type == "CONN_COMP":    
                    ocel_log = dataimport.load_ocel_json_xml("assets/uploaded_logs/" + uploaded_file, parameters={"execution_extraction":"connected_components"})
                else:
                    ocel_log = dataimport.load_ocel_json_xml("assets/uploaded_logs/" + uploaded_file, parameters={"execution_extraction":"leading_type", "leading_type":leading_obj})
            set_progress(("3","10"))

            # extract and store process executions as list, i.e. list of event ids within process execution
            ocel_process_executions = process_executions.get_process_executions(ocel_log)
            ocel_process_executions_list = process_executions.convert_process_executions_tolist(ocel_process_executions)
            
            set_progress(("5","10"))
            # extract and store ocel parameters
            ocel_df, _ = dataimport.ocel_to_df(ocel_log)
            object_types, num_events, num_activities, num_obj, activity_count, object_types_occurences = dataimport.get_summary(ocel_log, ocel_df)
            dict_params = {'object_types': object_types, 'num_events': num_events, 'num_activities':num_activities, 'num_objects':num_obj, 'activity_count':activity_count, 'object_type_occurences':object_types_occurences}
            
            set_progress(("9","10"))
            # encode ocel
            encoded_ocel = codecs.encode(pickle.dumps(ocel_log), "base64").decode()
            set_progress(("10","10"))

            return encoded_ocel, dict_params, ocel_process_executions_list, {'display':'block'}
    else:
        raise PreventUpdate

# load head of ocel into DataTable
@app.callback(Output("ocel-table", "children"), [Input("ocel_obj", "data"), Input("uploaded-logs", "value")], prevent_initial_call = False)
def on_upload_ocel_head(ocel_log, filename):
    triggered_id = ctx.triggered_id
    # if selected file is csv, read csv and display head
    if triggered_id == 'uploaded-logs':
        if filename is None:
            return dash.no_update
        elif filename.endswith("csv"):
            ocel_df = pd.read_csv("assets/uploaded_logs/" + filename)
            ocel_df = dataimport.remove_prefix_csv(ocel_df)
            ocel_df_head = ocel_df.head(5)
            return dbc.Table.from_dataframe(ocel_df_head, striped=True, bordered=True, hover=True)
        else:
            return dbc.Table.from_dataframe(dummy_df, striped=True, bordered=True, hover=True)
    
    # if file is already uploaded as ocel, load ocel object, transform to df and display head
    elif triggered_id == 'ocel_obj':
        if ocel_log != None:
            ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
            ocel_df, _ = dataimport.ocel_to_df(ocel_log)
            ocel_df_head = ocel_df.head(5)
            return dbc.Table.from_dataframe(ocel_df_head, striped=True, bordered=True, hover=True)
        else:
            return dash.no_update

# uncover csv parameter form, if selected file has csv extension
@app.callback([Output("csv-import", "style"), Output("id_name", "options"), Output("obj_names", "options"), Output("act_name", "options"), Output("time_name", "options"), Output("start_time_name", "options"), Output("id_name", "value"), Output("obj_names", "value"), Output("act_name", "value"), Output("time_name", "value")],Input("uploaded-logs", "value"), prevent_initial_call = True)
def on_selection_file(filename):
    if filename != None:
        # read head of csv and extract possible values for parameters, i.e. df column names
        if filename.endswith("csv"):
            ocel_csv = pd.read_csv("assets/uploaded_logs/" + filename, nrows=5)
            ocel_csv = dataimport.remove_prefix_csv(ocel_csv)
            column_names = ocel_csv.columns
            id_names_options = column_names
            obj_names_options = column_names
            act_names_options = column_names
            start_time_names_options = column_names
            time_names_options = column_names
            return {'display':'block'}, id_names_options, obj_names_options, act_names_options, time_names_options, start_time_names_options, column_names[0], None, column_names[0], column_names[0]
        else:
            return {'display':'none'}, [], [], [], [], [], [], None, None, None
    else:
        raise PreventUpdate