from transformers import BertModel, BertConfig, BertJapaneseTokenizer

from src.util import PRETRAINED_BERT_DIR
from src.load import get_text


def pull_main_from_livedoor_news(text):
    # 改行文字で分割してリストで返す
    lines = text.splitlines()
    # url, 日付、タイトル、本文を取得
    # url = lines[0]
    # datetime = lines[1]
    subject = lines[2]
    # 記事中の本文を1行にまとめる
    body = "".join(lines[3:])
    # タイトルと本文をまとめる
    text = subject + body

    return(text)


def clean_text(text):
    text = [sentence.strip() for sentence in text]  # 空白文字(スペースやタブ、改行)の削除
    text = list(filter(lambda line: line != '', text))
    text = ''.join(text)
    text = text.translate(str.maketrans(
        {'\n': '', '\t': '', '\r': '', '\u3000': ''}))  # 改行やタブ、全角スペースを消す

    return text


def tokenizer_512(input_text, tokenizer):
    """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
    return tokenizer.encode(input_text, truncation=True, max_length=512, pad_to_max_length=True, return_tensors='pt')[0]


class BertTransform():
    def __init__(self):
        self.model = BertModel.from_pretrained(str(PRETRAINED_BERT_DIR))
        self.config = BertConfig(str(PRETRAINED_BERT_DIR))
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(str(PRETRAINED_BERT_DIR))

        self.text_cleaning = ''
        self.token = []
        self.text_decode = []

    def __call__(self, text_path, phase):
        text = get_text(text_path)
        self.text_cleaning = clean_text(text)
        self.token = tokenizer_512(self.text_cleaning, self.tokenizer)
        self.text_decode = self.tokenizer.convert_ids_to_tokens(self.token.tolist())
        
        return self.token
