import os
import pickle
import json

from src.util import DATA_DIR_PATH


def dump_to_pickle(path, dump_obj):
    if (not path.parent.exists() and DATA_DIR_PATH == path.parent.parent):  # 無ければ
        os.makedirs(path.parent)
    
    with open(path, 'wb') as f:
        pickle.dump(dump_obj, f)


def dump_to_json(path, dump_obj):
    with open(path, 'w') as f:
        json.dump(dump_obj, f, indent=4)
