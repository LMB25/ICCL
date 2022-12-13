# Import necessary libraries 
from dash import html, dash_table, dcc, ctx
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from app import app
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import Dash, Trigger, ServersideOutput
from dash.exceptions import PreventUpdate
import os
import pickle
import codecs
from pathlib import Path
import ast
import time
import base64, io

from functions import dataimport, process_executions

# create empty Dataframe to display before any OCEL is uploaded
dummy_df = pd.DataFrame(columns=['event_id', 'activity', 'timestamp', 'object'])

# create empty DataTable
ocel_table = dbc.Table.from_dataframe(dummy_df, striped=True, bordered=True, hover=True, id="ocel_table_head")

# create Dropdown for files
file_dropdown = dcc.Dropdown(id='file-dropdown')

# Define Store object for csv parameters
csv_store = dcc.Store(id='csv-params', storage_type='local')

# csv import parameters form
csv_import = html.Div([
        csv_store,
        html.H5("Please specify the necessary parameters for OCEL csv import"),
        dbc.Row([
            dbc.Col(html.P("Enter object names: ")),
            dbc.Col(dbc.Input(id='obj_names', placeholder='["application", "offer"]'))
        ]),
        dbc.Row([
            dbc.Col(html.P("Enter attribute names: ")),
            dbc.Col(dbc.Input(id='val_names', placeholder='[]'))
        ]),
        dbc.Row([
            dbc.Col(html.P("Enter column name of event's activity: ")),
            dbc.Col(dbc.Input(id='act_name', placeholder='event_activity'))
        ]),
        dbc.Row([
            dbc.Col(html.P("Enter column name of event's timestamp: ")),
            dbc.Col(dbc.Input(id='time_name', placeholder='event_timestamp'))
        ]),
        dbc.Row([
            dbc.Col(html.P("Enter seperator: ")),
            dbc.Col(dbc.Input(id='sep', placeholder=','))
        ]),
        html.Br(),
        dbc.Row([
            dbc.Button("Parse csv Parameters", color="warning", id="parse-csv", className="me-2", n_clicks=0)
        ])], id='csv-import', style={'display': 'none'})

# Define the page layout
layout = dbc.Container([
        html.Center(html.H1("Import Data")),
        html.Hr(),
        dcc.Upload(id="drag-drop-field", children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                   style={"width": "100%", "height": "60px", "lineHeight": "60px", "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center", "margin": "10px",}
                   ),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.P("Insert Path to OCEL")), 
            dbc.Col(dbc.Input(id="path", value=os.path.dirname(os.path.realpath(__file__)), type="text", persistence = False)),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Button("Search", id="enable-path", className="me-2", n_clicks=0, disabled=False),
                ]),
        html.Div("Path successfully searched.", style={'display':'none'}, id='folder-search-result'),
        html.Br(),
        dbc.Row([
            dbc.Col(html.P("Select OCEL File, allowed extensions: jsonocel, xmlocel, csv")), 
            dbc.Col(file_dropdown)
        ]),
        html.Br(),
        csv_import,
        html.Div("Parameters successfully parsed.", style={'display':'none'}, id='success-parse-csv'),
        html.Br(),
        dbc.Row([
            dbc.Button("Upload", id="upload-button", className="me-2", n_clicks=0, disabled=True),
                ]), 
        html.Div("OCEL successfully uploaded.", style={'display':'none'}, id='success-upload-ocel'),
        html.Br(),
        html.Div([
            ocel_table
            ], id='ocel-table')
    ])

# callback for path-files store
@app.callback([Output("folder-selection", "data"), Output("folder-search-result", "style")], [State("path", "value")], [Input("enable-path", "n_clicks")], prevent_initial_call=True)
def on_get_filepath(value, n):
    if n > 0:
        if value is None:
            raise PreventUpdate
        else:
            ext = ["jsonocel", "xmlocel", "csv"]
            files = []
            # list files that have one of the extensions
            for filename in os.listdir(value):
                if filename.endswith(tuple(ext)):
                    files.append(filename)
            # convert to df
            df = pd.DataFrame(files, columns=['files'])
            # convert to dictionary
            dict_df = df.to_dict()
        return dict_df, {'display':'block'}
    else:
        raise PreventUpdate

# load filenames into dropdown
@app.callback(Output("file-dropdown", "options"), Input("folder-selection", "data"), prevent_initial_call=True)
def on_selection_folder(files):
    if files is None:
        raise PreventUpdate
    else:
        filenames = []
        file_dict = files['files']
        for enumeration in file_dict.keys():
            filenames.append(file_dict[enumeration])
        df = pd.DataFrame(filenames, columns=['files'])
        options=[{'label':file, 'value':file} for file in df['files'].unique()]
        return options

# load csv parameters into store
@app.callback([Output("csv-params", "data"), Output("success-parse-csv", "style")], [State("obj_names", "value"), State("val_names", "value"), State("act_name", "value"), State("time_name", "value"), State("sep", "value"), State("file-dropdown", "value"), State("drag-drop-field", "filename")], Input("parse-csv",  "n_clicks"), prevent_initial_call=True)
def on_upload_csv(obj_name, val_name, act_name, time_name, sep, filename, drag_drop_filename, n):
    if (filename != None and filename.endswith("csv")) or drag_drop_filename.endswith("csv"):
        if obj_name is None:
            obj_name = []
        elif obj_name.startswith("["):
            # convert string into list
            obj_name = ast.literal_eval(obj_name)
        else:
            obj_name = list(obj_name)
        if val_name is None:
            val_name = []
        elif val_name.startswith("["):
            # convert string into list
            val_name = ast.literal_eval(val_name)
        else:
            val_name = list(val_name)
        if act_name is None:
            act_name = "event_activity"
        if time_name is None:
            time_name = "event_timestamp"
        if sep is None:
            sep = ","
        params = {"obj_names":obj_name,
                "val_names":val_name,
                "act_name":act_name,
                "time_name":time_name,
                "sep":sep}
        return params, {'display':'block'}
    else:
        return {}, {'display':'none'}

# enable upload button
@app.callback(Output("upload-button", "disabled"), [Input("csv-params", "data"), Input("file-dropdown", "value"), Input("drag-drop-field", "contents")], prevent_initial_call = True)
def on_file_selection(csv_params_parsed, selected_file, drag_drop_content):
    if selected_file != None: 
        if (csv_params_parsed != None) or (selected_file.endswith("csv") == False):
            return False
    elif drag_drop_content is not None:
        return False 
    else:
        return True

# load and store ocel, extract and store parameters, uncover 'success' div
@app.callback([ServersideOutput("ocel_obj", "data"), Output("param-store", "data"), ServersideOutput("execution-store", "data"), Output("success-upload-ocel", "style")], [State("file-dropdown", "value"), State("path", "value"), State("csv-params", "data"),State("drag-drop-field", "contents"),State("drag-drop-field", "filename"),], [Trigger("upload-button",  "n_clicks")], memoize=True)
def on_upload_ocel_path(selected_file, selected_dir, csv_params, drag_drop_content, drag_drop_filename, n):
    time.sleep(1)
    if selected_file is None and drag_drop_content is None:
        raise PreventUpdate
    else:
        # use different load function w.r.t file extension
        if selected_file!=None and selected_file.endswith("csv"):
            ocel_log = dataimport.load_ocel_csv(os.path.join(selected_dir, selected_file), csv_params)
        elif drag_drop_filename!=None and drag_drop_filename.endswith("csv"):
            ocel_log = dataimport.load_ocel_csv_drag_droph(drag_drop_content, csv_params)
        else:
            if drag_drop_content!=None and drag_drop_filename.endswith("jsonocel"):
                ocel_log = dataimport.load_ocel_drag_drop(drag_drop_content)
            else:
                ocel_log = dataimport.load_ocel_json_xml(os.path.join(selected_dir, selected_file))

        # remove any existing discovered nets, if exist
        [f.unlink() for f in Path("/imgs").glob("*") if f.is_file()] 

        # extract and store process executions as list, i.e. list of event ids within process execution
        ocel_process_executions = process_executions.get_process_executions(ocel_log)
        ocel_process_executions_list = process_executions.convert_process_executions_tolist(ocel_process_executions)

        # extract and store ocel parameters
        ocel_df, _ = dataimport.ocel_to_df_params(ocel_log)
        object_types, num_events, num_activities, num_obj, activity_count, object_types_occurences = dataimport.get_summary(ocel_log, ocel_df)
        dict_params = {'object_types': object_types, 'num_events': num_events, 'num_activities':num_activities, 'num_objects':num_obj, 'activity_count':activity_count, 'object_type_occurences':object_types_occurences}

        # encode ocel
        encoded_ocel = codecs.encode(pickle.dumps(ocel_log), "base64").decode()

        return encoded_ocel, dict_params, ocel_process_executions_list, {'display':'block'}

# load head of ocel df
@app.callback(Output("ocel-table", "children"), State("path", "value"), [Input("ocel_obj", "data"), Input("file-dropdown", "value"), Input("drag-drop-field", "filename"), State("drag-drop-field", "contents")], prevent_initial_call = True)
def on_upload_ocel_head(selected_dir, ocel_log, filename, drag_drop_filename, drag_drop_contents):
    triggered_id = ctx.triggered_id
    # if selected file is csv, read csv and display head
    if triggered_id == 'file-dropdown':
        if filename is None:
            raise PreventUpdate
        elif filename.endswith("csv"):
            ocel_df = pd.read_csv(os.path.join(selected_dir, filename))
            ocel_df_head = ocel_df.head(10)
            return dbc.Table.from_dataframe(ocel_df_head, striped=True, bordered=True, hover=True)
        else:
            return ocel_table
        
     # if selected file (uploaded with the drag-drop-field) is csv, read csv and display head
    if triggered_id == 'drag-drop-field':
        if drag_drop_filename is None:
            raise PreventUpdate
        elif drag_drop_filename.endswith("csv"):
            content_type, content_string = drag_drop_contents.split(',')
            decoded = base64.b64decode(content_string)
            ocel_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            ocel_df_head = ocel_df.head(10)
            return dbc.Table.from_dataframe(ocel_df_head, striped=True, bordered=True, hover=True)
        else:
            return ocel_table    
    
    # if file is already uploaded as ocel, load ocel object, transform to df and display head
    elif triggered_id == 'ocel_obj':
        if ocel_log != None:
            ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
            ocel_df, _ = dataimport.ocel_to_df_params(ocel_log)
            ocel_df_head = ocel_df.head(5)
            return dbc.Table.from_dataframe(ocel_df_head, striped=True, bordered=True, hover=True)

# uncover csv parameter form, if selected file has csv extension
@app.callback(Output("csv-import", "style"), [Input("file-dropdown", "value"), Input("drag-drop-field", "filename")], prevent_initial_call = True)
def on_selection_file(filename, drag_drop_filename):
    if filename != None:
        if filename.endswith("csv"):
            return {'display':'block'}
        else:
            return {'display':'none'}
        
    elif drag_drop_filename != None:
        if drag_drop_filename.endswith("csv"):
            return {'display':'block'}
        else:
            return {'display':'none'}
    else:
        dash.no_update