from datetime import datetime
import json


def round_floats(o):
    if isinstance(o, float):
        return round(o, 5)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def log(method, dic):
    fname = method + datetime.now().strftime("_%Y-%m-%d_%H:%M") + '.txt'
    with open('./logs/' + fname, 'w') as f:
        json.dump(round_floats(dic), f)
