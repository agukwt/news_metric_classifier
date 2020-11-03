from transformers import BertConfig, BertModel, BertJapaneseTokenizer

from src.util import PRETRAINED_BERT_DIR


def load_tohoku_unv_bert_infomation() -> dict:
    """東北大学の学習済みデータをロード

    Returns
    -------
    dict
        {'model': model,
         'config': config,
         'tokenizer': tokenizer
        }
    """

    model = BertModel.from_pretrained(str(PRETRAINED_BERT_DIR))
    config = BertConfig(str(PRETRAINED_BERT_DIR))
    tokenizer = BertJapaneseTokenizer.from_pretrained(str(PRETRAINED_BERT_DIR))

    return {'model': model, 'config': config, 'tokenizer': tokenizer}
