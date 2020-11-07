import pickle
from transformers import BertConfig, BertModel, BertJapaneseTokenizer

from src.util import PRETRAINED_BERT_DIR
# from src.util import CORPUS_DIR_PATH


def load_from_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
