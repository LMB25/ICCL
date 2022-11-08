import pm4py
import ocpa 

def load_ocel(path):
    ocel_file = pm4py.read_ocel(path)
    return ocel_file

def get_ocel_table(log):
    ocel_table = log.get_extended_table()
    return ocel_table
