"""
    user.py
    ~~~
    This module provides all API for tracks

    :auther: Alexander Z Wang
"""

import json


def get_hidden_features(catalog, track_id, options="CF"):

    if options == "CF":
        with open(catalog) as data_file:
            data = json.load(data_file)

        return data[track_id]

    return []
