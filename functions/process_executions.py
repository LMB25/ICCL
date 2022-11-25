

def get_process_executions(ocel):
    # default: connected components
    return ocel.process_executions

def convert_process_executions_tolist(process_executions):
    process_executions_list = [list(process_ex) for process_ex in process_executions]
    return process_executions_list

def get_events_process_exection(ocel, id):
    return ocel.process_executions[id]

def get_objects_process_execution(ocel, id):
    return ocel.process_execution_objects[id]

def get_process_execution_graph(ocel, id):
    ocel_process_executions = get_process_executions(ocel)
    return ocel.graph.eog.subgraph(ocel_process_executions[id])

def get_variant_process_executions(ocel):
    return ocel.variants_dict()