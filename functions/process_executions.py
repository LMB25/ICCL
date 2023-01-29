# functions related to process executions

# get list of process executions 
def get_process_executions(ocel):
    return ocel.process_executions

# get list of lists of event ids of process executions
def convert_process_executions_tolist(process_executions):
    process_executions_list = [list(process_ex) for process_ex in process_executions]
    return process_executions_list

# get events of certain process execution
def get_events_process_exection(ocel, id):
    return ocel.process_executions[id]

# get objects of certain process execution
def get_objects_process_execution(ocel, id):
    return ocel.process_execution_objects[id]

# get process execution graph of certain process execution
def get_process_execution_graph(ocel, id):
    graph = ocel.get_process_execution_graph(id)
    return graph
