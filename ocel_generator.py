import json
import re
from datetime import datetime, timedelta

from ocpa.objects.log.importer.ocel import factory as ocel_import_factory

# specify all involved objects and their corresponding object types
OBJECTS = {0: "o1", 1: "i1", 2: "i2", 3: "o2", 4: "i3"}
OBJECT_TYPES = {"o1": "order", "i1": "item", "i2": "item", "o2": "order", "i3": "item"}
GENERATED_OBJECTS = {}  # leave empty

# specify each trace as a list of tuples of activity and corresponding object ids (from OBJECTS dict)
ACTIVITIES = [
    [
        ("place_order", [0, 1, 2]),
        ("send_invoice", [1]),
        ("send_invoice", [2]),
        ("pick_item", [2]),
        ("pick_item", [1]),
        ("pay_order", [0]),
        ("send_delivery", [0, 1, 2]),
        ("delivery_received", [0, 1, 2]),
        ("archive_order", [0]),
        ("receive_review", [1]),
        ("receive_review", [2]),
    ],
    [
        ("place_order", [3, 4]),
        ("send_invoice", [4]),
        ("item_unavailable", [4]),
        ("cancel_order", [3, 4]),
    ],
]


def create_object_instance(obj_nr, n):
    obj = OBJECTS[obj_nr]
    obj_type = OBJECT_TYPES[obj]

    number_of_objects = len([o for o in OBJECT_TYPES.values() if o == obj_type])

    groups = re.match(r"([a-z]+)([0-9]+)", obj).groups()

    new_obj = groups[0] + str(int(groups[1]) + n * number_of_objects)

    GENERATED_OBJECTS[new_obj] = obj_type

    return new_obj


def create_activity_array(number_of_traces):
    activities = []
    for n in range(0, number_of_traces):
        activities = activities + [
            (act[0], [create_object_instance(i, n) for i in act[1]])
            for act in ACTIVITIES[0] + ACTIVITIES[1]
        ]

    return activities


def create_json(path, repeat_trace):

    activity_array = create_activity_array(repeat_trace)

    times = []
    d = datetime(2022, 1, 1, 12, 0, 0, 0)
    for i in range(0, len(activity_array)):
        times.append(datetime.strftime(d, "%Y-%m-%d %H:%M:%S"))
        d = d + timedelta(minutes=1)

    events = {
        f"{i}": {
            "ocel:activity": a[0],
            "ocel:timestamp": times[i],
            "ocel:omap": a[1],
            "ocel:vmap": {},
        }
        for i, a in enumerate(activity_array)
    }

    log = {
        "ocel:global-log": {
            "ocel:attribute-names": [],
            "ocel:object-types": list(set(OBJECT_TYPES.values())),
            "ocel:version": ["1.0"],
            "ocel:ordering": ["timestamp"],
        },
        "ocel:global-event": {},
        "ocel:global-object": {},
        "ocel:events": events,
        "ocel:objects": {
            f"{o}": {"ocel:type": ot, "ocel:ovmap": {}}
            for o, ot in (OBJECT_TYPES.items() | GENERATED_OBJECTS.items())
        },
    }

    with open(path, "w") as f:
        json.dump(log, f, indent=4)


create_json("logs/gen_log.jsonocel", 2)

ocel = ocel_import_factory.apply("logs/gen_log.jsonocel")

print(ocel)
