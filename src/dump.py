import pickle
import os

from src.util import DATA_DIR_PATH


def dump_to_pickle(path, dump_obj):
    if (not path.parent.exists() and DATA_DIR_PATH == path.parent.parent):  # 無ければ
        os.makedirs(path.parent)
    
    with open(path, 'wb') as f:
        pickle.dump(dump_obj, f)
