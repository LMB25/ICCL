import pandas as pd
from ocpa.objects.log.importer.csv import factory as ocel_import_factory_csv
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ast import literal_eval
import math
import ocpa.objects.log.util.misc as import_helper

# remove prefix from csv OCEL columns
def remove_prefix_csv(df):
    columns = df.columns
    new_columns = []
    for column in columns:
        suffix = column.rsplit(':', 1)[-1]
        new_columns.append(suffix)
    df.columns = new_columns
    return df

# load ocel from csv format (given a path and parameters)
def load_ocel_csv(path, parameters):
    ocel_file = ocel_import_factory_csv.apply(file_path=path, variant="to_ocel", parameters = parameters)
    return ocel_file

# load ocel from jsnocel or xmlocel format
def load_ocel_json_xml(path, parameters):
    if path.endswith("jsonocel"):
        ocel_file = ocel_import_factory.apply(path, parameters=parameters)
    elif path.endswith("xmlocel"):
        ocel_file = ocel_import_factory.apply(path, parameters=parameters)
    else:
        error_msg = "not a valid extension"
        return error_msg
    return ocel_file

# preprocess df of OCEL csv
def preprocess_csv(df, parameters=None):
    if parameters is None:
        raise ValueError("Specify parsing parameters")
    obj_cols = parameters['obj_names']

    def _eval(x):
        if x == 'set()':
            return []
        elif type(x) == float and math.isnan(x):
            return []
        else:
            return literal_eval(x)

    if obj_cols:
        for c in obj_cols:
            df[c] = df[c].apply(_eval)

    df['activity'] = df[parameters['act_name']]
    if not parameters['act_name'] == 'activity':
        del df[parameters['act_name']]
    df['timestamp'] = df[parameters['time_name']]
    if not parameters['time_name'] == 'timestamp':
        del df[parameters['time_name']]
    if "start_timestamp" in parameters:
        df['start_timestamp'] = df[parameters['start_timestamp']]
        del df[parameters['start_timestamp']]
    else:
        df['start_timestamp'] = df['timestamp']
    #### change to ocpa version
    df['event_id'] = df[parameters['id_name']].values
    # check if event id column is numeric
    try:
        pd.to_numeric(df['event_id'])
    except:
        # create a numeric event id column
        df['event_id'] = [i for i in range(0, len(df))]

    #df = df.drop(columns=[parameters['id_name']])


    rename_dict = {}
    for col in [x for x in df.columns if not x.startswith("event_")]:
        if col not in obj_cols:
            rename_dict[col] = 'event_' + col
    df.rename(columns=rename_dict, inplace=True)

    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df = df.dropna(subset=["event_id"])
    df["event_id"] = df["event_id"].astype(str)

    #rename dict params
    parameters['act_name'] = "event_activity"
    parameters['time_name'] = "event_timestamp"
    parameters['id_name'] = "event_id"

    #convert list objects to str
   # for obj_col in parameters['obj_names']:
   #     df[obj_col] = df[obj_col].apply(', '.join)

    return df, parameters

# construct ocel from DataFrame
def df_to_ocel(df, parameters):
    
    df, parameters = preprocess_csv(df, parameters)
    ocel = import_helper.copy_log_from_df(df, parameters)
    
    return ocel

# convert ocel to DataFrame
def ocel_to_df(ocel, return_obj_df=False, parameters=None):
    if parameters is None:
        parameters = {}
    if 'return_obj_df' in parameters:
        return_obj_df = parameters['return_obj_df']
    else:
        return_obj_df = True

    prefix = "ocel:"

    objects = ocel.obj.raw.objects
    events = ocel.obj.raw.events

    obj_type = {}
    for obj in objects:
        obj_type[objects[obj].id] = objects[obj].type
    eve_stream = []
    for ev in events:
        # print(events[ev])
        new_omap = {}
        for obj in events[ev].omap:
            typ = obj_type[obj]
            if not typ in new_omap:
                new_omap[typ] = set()
            new_omap[typ].add(obj)
        for typ in new_omap:
            new_omap[typ] = list(new_omap[typ])
        el = {}
        el["event_id"] = events[ev].id
        el["event_activity"] = events[ev].act
        el["event_timestamp"] = events[ev].time
        for k2 in events[ev].vmap:
            if k2.startswith("event_"):
                el[k2] = events[ev].vmap[k2]
            else:
                el["event_" + k2] = events[ev].vmap[k2]
        for k2 in new_omap:
            el[k2] = new_omap[k2]
        eve_stream.append(el)

    obj_stream = []

    eve_df = pd.DataFrame(eve_stream)
    # if an object is empty for an event, replace them with empty list []
    for col in eve_df.columns:
        if 'event' not in col:
            eve_df[col] = eve_df[col].apply(
                lambda d: d if isinstance(d, list) else [])
    obj_df = pd.DataFrame(obj_stream)

    eve_df.type = "succint"

    if return_obj_df or (return_obj_df is None and len(obj_df.columns) > 1):
        return eve_df, obj_df
    return eve_df, []

# get object types of OCEL
def get_ocel_object_types(ocel):
    object_types = ocel.object_types
    return object_types

# extract summary of OCEL
def get_summary(ocel, ocel_df):
    object_types = get_ocel_object_types(ocel)
    num_events = len(ocel_df)
    num_activities = len(ocel_df['event_activity'].unique())
    activity_count = ocel_df['event_activity'].value_counts().to_dict()
    object_types_occurences = dict.fromkeys(object_types, 0)

    num_obj = 0
    # find number of objects for each object type
    for obj in object_types:
        df_obj = ocel_df[[obj]]
        df_obj = df_obj.explode(obj)
        df_obj = df_obj.dropna()
        num_df_obj = len(df_obj[obj].unique())
        object_types_occurences[obj] += num_df_obj
        num_obj += num_df_obj

    return object_types, num_events, num_activities, num_obj, activity_count, object_types_occurences
