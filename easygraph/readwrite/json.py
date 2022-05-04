#!/usr/bin/env python3

import easygraph as eg


# write to a json like this: {"class": "Graph", "data": json formatted __dict__}
def write_json(file_name, obj):
    import json
    json_dict = {}
    json_dict["class"] = obj.__class__.__name__
    json_dict["data"] = obj.__dict__
    with open(file_name, 'w') as f:
        json.dump(json_dict, f, indent=4)


def read_json(file_name):
    import json
    with open(file_name, 'r') as f:
        json_dict = json.load(f)
    class_name = json_dict["class"]
    # create the instance with easygraph.class_name()
    class_ = getattr(eg, class_name)
    obj = class_()
    # set the attributes
    obj.__dict__ = json_dict["data"]
    return obj
