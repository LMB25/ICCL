from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle
from ocpa.objects.log.variants.obj import Event, Obj, ObjectCentricEventLog
from ocpa.objects.log.util.param import JsonParseParameters
from ocpa.objects.log.variants.obj import Event, Obj, ObjectCentricEventLog, MetaObjectCentricData, RawObjectCentricData
from typing import Dict, List, Any, Union
from collections import OrderedDict
import json
from ocpa.objects.log.ocel import OCEL
import ocpa.objects.log.converter.factory as convert_factory
#from functions import dataimport

def ocel_to_jsonobj(ocel: OCEL, parameters=None):
    cfg = JsonParseParameters(None)
    meta = ocel.obj.meta
    raw = ocel.obj.raw
    export = dict()
    export[cfg.log_params["meta"]] = dict()
    export[cfg.log_params["meta"]][cfg.log_params["attr_names"]] = meta.attr_names
    export[cfg.log_params["meta"]][cfg.log_params["obj_types"]] = meta.obj_types
    export[cfg.log_params["meta"]][cfg.log_params["version"]] = "1.0"
    export[cfg.log_params["meta"]][cfg.log_params["ordering"]] = "timestamp"
    events = {}
    for event in raw.events.values():
        events[event.id] = {}
        events[event.id][cfg.event_params["act"]] = event.act
        events[event.id][cfg.event_params["time"]] = event.time.isoformat()
        events[event.id][cfg.event_params["omap"]] = event.omap
        events[event.id][cfg.event_params["vmap"]] = event.vmap

    objects = {}
    for obj in raw.objects.values():
        objects[obj.id] = {}
        objects[obj.id][cfg.obj_params["type"]] = obj.type
        objects[obj.id][cfg.obj_params["ovmap"]] = obj.ovmap

    export[cfg.log_params["events"]] = events
    export[cfg.log_params["objects"]] = objects
    #json.dump(export, f, ensure_ascii=False, indent=4, default=str)
    return export


def json_to_ocel(data: Dict[str, Any]) -> ObjectCentricEventLog:
    cfg = JsonParseParameters()
    # parses the given dict
    events = parse_events(data[cfg.log_params['events']], cfg)
    objects = parse_objects(data[cfg.log_params['objects']], cfg)
    # Uses the last found value type
    attr_events = {v:
                   str(type(events[eid].vmap[v])) for eid in events
                   for v in events[eid].vmap}
    attr_objects = {v:
                    str(type(objects[oid].ovmap[v])) for oid in objects
                    for v in objects[oid].ovmap
                    }
    attr_types = list({attr_events[v] for v in attr_events}.union(
        {attr_objects[v] for v in attr_objects}))
    attr_typ = {**attr_events, **attr_objects}
    act_attr = {}
    for eid, event in events.items():
        act = event.act
        if act not in act_attr:
            act_attr[act] = {v for v in event.vmap}
        else:
            act_attr[act] = act_attr[act].union({v for v in event.vmap})
    for act in act_attr:
        act_attr[act] = list(act_attr[act])
    meta = MetaObjectCentricData(attr_names=data[cfg.log_params['meta']][cfg.log_params['attr_names']],
                                 obj_types=data[cfg.log_params['meta']
                                                ][cfg.log_params['obj_types']],
                                 attr_types=attr_types,
                                 attr_typ=attr_typ,
                                 act_attr=act_attr,
                                 attr_events=list(attr_events.keys()))
    data = ObjectCentricEventLog(
        meta, RawObjectCentricData(events, objects))
    
    # added conversion of ObjectCentricEventLog to OCEL object
    ocel, _ = convert_factory.apply(data, variant='json_to_mdl')
    return ocel


def parse_events(data: Dict[str, Any], cfg: JsonParseParameters) -> Dict[str, Event]:
    # Transform events dict to list of events
    act_name = cfg.event_params['act']
    omap_name = cfg.event_params['omap']
    vmap_name = cfg.event_params['vmap']
    time_name = cfg.event_params['time']
    events = {}
    for item in data.items():
        events[item[0]] = Event(id=item[0],
                                act=item[1][act_name],
                                omap=item[1][omap_name],
                                vmap=item[1][vmap_name],
                                time=item[1][time_name])
        if "start_timestamp" not in item[1][vmap_name]:
            events[item[0]].vmap["start_timestamp"] = item[1][time_name]
        else:
            events[item[0]].vmap["start_timestamp"] = events[item[0]].vmap["start_timestamp"]
    sorted_events = sorted(events.items(), key=lambda kv: kv[1].time)
    return OrderedDict(sorted_events)


def parse_objects(data: Dict[str, Any], cfg: JsonParseParameters) -> Dict[str, Obj]:
    # Transform objects dict to list of objects
    type_name = cfg.obj_params['type']
    ovmap_name = cfg.obj_params['ovmap']
    objects = {item[0]: Obj(id=item[0],
                            type=item[1][type_name],
                            ovmap=item[1][ovmap_name])
               for item in data.items()}
    return objects

