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
from components import explanation_texts, input_forms

# create empty Dataframe to display before any OCEL is uploaded
dummy_df = pd.DataFrame(columns=['event_id', 'activity', 'timestamp', 'object'])

# create empty DataTable
ocel_table = dbc.Table.from_dataframe(dummy_df, striped=True, bordered=True, hover=True, id="ocel_table_head")

# create Dropdown for files
file_dropdown = dcc.Dropdown(id='file-dropdown')

# create store for csv params
csv_params = dcc.Store(id='csv-params', storage_type='local')

# create html div for leading object dropdown
leading_object_div = html.Div([ 
                                dcc.Dropdown(placeholder='Select leading object type', id='leading-object', style={'display': 'block', 'width':'80%'})
                                ])

# create selection for leading object type oder connected component process execution extraction
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
        dcc.Upload(id="drag-drop-field", children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                   style={"width": "100%", "height": "60px", "lineHeight": "60px", "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center", "margin": "10px",}
                   ),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.P("Insert Path to OCEL"), width=3), 
            dbc.Col(dbc.Input(id="path", value=os.path.dirname(os.path.realpath(__file__)), type="text", persistence = False), width=7),
            dbc.Col(dbc.Button("Search", id="enable-path", className="me-2", n_clicks=0, disabled=False))
        ]),
        html.Br(),
        html.Div("Path successfully searched.", style={'display':'none'}, id='folder-search-result'),
        html.Br(),
        dbc.Row([
            dbc.Col(html.P("Select OCEL File, allowed extensions: jsonocel, xmlocel, csv")), 
            dbc.Col(file_dropdown)
        ]),
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
@app.callback([Output("csv-params", "data"), Output("success-parse-csv", "style")], [State("obj_names", "value"), State("act_name", "value"), State("start_time_name", "value"), State("time_name", "value"), State("id_name", "value"), State("file-dropdown", "value"), State("drag-drop-field", "filename")], Input("parse-csv",  "n_clicks"), prevent_initial_call=True)
def on_upload_csv(obj_name, act_name, start_time_name, time_name, id_name, filename, drag_drop_filename, n):
    if (filename != None and filename.endswith("csv")) or drag_drop_filename.endswith("csv"):
        sep = ","
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
    else:
        return {}, {'display':'none'}

# load possible leading object types into dropdown
@app.callback(Output("leading-object", "options"), [Input("csv-params","data"), Input("process-extraction-type", "value")], [State("path", "value"), State("file-dropdown", "value"), State("drag-drop-field", "contents")], prevent_initial_call=True)
def on_parse_params(csv_params, process_ex_type, path, filename, drag_drop_content):
    #if (filename.endswith("csv")) and (csv_params != None):
    if csv_params != None:
        options = csv_params['obj_names']
        return options 
    #elif (not filename.endswith("csv")):
    else:
        if process_ex_type == "CONN_COMP":
            return [] 
        else: 
            # load OCEL either from path and filename combination or drag and drop field
            if filename != None:
                ocel_log = dataimport.load_ocel_json_xml(os.path.join(path, filename), parameters=None)
            else: 
                ocel_log = dataimport.load_ocel_drag_drop(drag_drop_content)
            # extract object types from OCEL
            object_types = dataimport.get_ocel_object_types(ocel_log)
            return object_types


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
@app.long_callback(output=(
    ServersideOutput("ocel_obj", "data"), 
    Output("param-store", "data"), 
    ServersideOutput("execution-store", "data"), 
    Output("success-upload-ocel", "style")
    ),inputs=(
        State("file-dropdown", "value"), 
        State("path", "value"), 
        State("csv-params", "data"),
        State("drag-drop-field", "contents"),
        State("drag-drop-field", "filename"),
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
    memoize=True)
def on_upload_ocel_path(set_progress, selected_file, selected_dir, csv_params, drag_drop_content, drag_drop_filename, process_extr_type, leading_obj, n):
    set_progress(("0","10"))
    time.sleep(1)
    if selected_file is None and drag_drop_content is None:
        raise PreventUpdate
    else:
        # use different load function w.r.t file extension
        if selected_file!=None and selected_file.endswith("csv"):
            #ocel_log = dataimport.load_ocel_csv(os.path.join(selected_dir, selected_file), csv_params)
            ocel_df = pd.read_csv(os.path.join(selected_dir, selected_file))
            ocel_df = dataimport.remove_prefix_csv(ocel_df)
            if process_extr_type == "CONN_COMP":
                ocel_log = dataimport.df_to_ocel(ocel_df, csv_params)
            else:
                # add leading object for process execution extraction
                csv_params["execution_extraction"] = "leading_type"
                csv_params["leading_type"] = leading_obj
                ocel_log = dataimport.df_to_ocel(ocel_df, csv_params)
        elif drag_drop_filename!=None and drag_drop_filename.endswith("csv"):
            ocel_log = dataimport.load_ocel_csv_drag_drop(drag_drop_content, csv_params)
        else:
            if drag_drop_content!=None and drag_drop_filename.endswith("jsonocel"):
                ocel_log = dataimport.load_ocel_drag_drop(drag_drop_content)
            else:
                if process_extr_type == "CONN_COMP":    
                    ocel_log = dataimport.load_ocel_json_xml(os.path.join(selected_dir, selected_file), parameters={"execution_extraction":"connected_components"})
                else:
                    ocel_log = dataimport.load_ocel_json_xml(os.path.join(selected_dir, selected_file), parameters={"execution_extraction":"leading_type", "leading_type":leading_obj})
        set_progress(("3","10"))
        # remove any existing discovered nets, if exist
        [f.unlink() for f in Path("/imgs").glob("*") if f.is_file()] 

        # extract and store process executions as list, i.e. list of event ids within process execution
        ocel_process_executions = process_executions.get_process_executions(ocel_log)
        ocel_process_executions_list = process_executions.convert_process_executions_tolist(ocel_process_executions)
        
        set_progress(("5","10"))
        # extract and store ocel parameters
        ocel_df, _ = dataimport.ocel_to_df_params(ocel_log)
        object_types, num_events, num_activities, num_obj, activity_count, object_types_occurences = dataimport.get_summary(ocel_log, ocel_df)
        dict_params = {'object_types': object_types, 'num_events': num_events, 'num_activities':num_activities, 'num_objects':num_obj, 'activity_count':activity_count, 'object_type_occurences':object_types_occurences}
        
        set_progress(("9","10"))
        # encode ocel
        encoded_ocel = codecs.encode(pickle.dumps(ocel_log), "base64").decode()
        set_progress(("10","10"))

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
            ocel_df = dataimport.remove_prefix_csv(ocel_df)
            ocel_df_head = ocel_df.head(5)
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
            ocel_df = dataimport.remove_prefix_csv(ocel_df)
            ocel_df_head = ocel_df.head(5)
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
@app.callback([Output("csv-import", "style"), Output("id_name", "options"), Output("obj_names", "options"), Output("act_name", "options"), Output("time_name", "options"), Output("start_time_name", "options"), Output("id_name", "value"), Output("obj_names", "value"), Output("act_name", "value"), Output("time_name", "value")], [Input("file-dropdown", "value"), Input("drag-drop-field", "filename")], [State("drag-drop-field", "contents"), State("path", "value")], prevent_initial_call = True)
def on_selection_file(filename, drag_drop_filename, drag_drop_contents, selected_dir):
    if filename != None:
        if filename.endswith("csv"):
            ocel_csv = pd.read_csv(os.path.join(selected_dir, filename), nrows=5)
            ocel_csv = dataimport.remove_prefix_csv(ocel_csv)
            column_names = ocel_csv.columns
            id_names_options = column_names
            obj_names_options = column_names
            act_names_options = column_names
            start_time_names_options = column_names
            time_names_options = column_names
            return {'display':'block'}, id_names_options, obj_names_options, act_names_options, time_names_options, start_time_names_options, column_names[0], None, column_names[0], column_names[0]
        else:
            return {'display':'none'}, [], [], [], [], [], None, None, None, None
        
    elif drag_drop_filename != None:
        if drag_drop_filename.endswith("csv"):
            content_type, content_string = drag_drop_contents.split(',')
            decoded = base64.b64decode(content_string)
            ocel_csv = pd.read_csv(io.StringIO(decoded.decode('utf-8')), nrows=5)
            ocel_csv = dataimport.remove_prefix_csv(ocel_csv)
            column_names = ocel_csv.columns
            id_names_options = column_names
            obj_names_options = column_names
            act_names_options = column_names
            start_time_names_options = column_names
            time_names_options = column_names
            return {'display':'block'}, id_names_options, obj_names_options, act_names_options, time_names_options, start_time_names_options, column_names[0], None, column_names[0], column_names[0]
        else:
            return {'display':'none'}, [], [], [], [], [], None, None, None, None
    else:
        dash.no_update