import pickle


def load_from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
