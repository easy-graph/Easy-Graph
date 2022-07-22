#!/usr/bin/env python3


from __future__ import annotations


def read_pickle(file_name):
    import pickle

    with open(file_name, "rb") as f:
        return pickle.load(f)


def write_pickle(file_name, obj):
    import pickle

    with open(file_name, "wb") as f:
        pickle.dump(obj, f)
