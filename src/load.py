import pickle


def load_from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
