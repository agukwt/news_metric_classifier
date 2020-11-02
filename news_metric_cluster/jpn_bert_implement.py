import torch
from transformers import BertJapaneseTokenizer
from pathlib import Path


BERT_MODEL_DIR = 'BERT-base_mecab-ipadic-bpe-32k_whole-word-mask'

# GPU環境確認
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

this_dir = Path(__file__)
wk_dir = this_dir.parent.parent
bert_dir = wk_dir.joinpath('data').joinpath(BERT_MODEL_DIR)

tokenizer = BertJapaneseTokenizer.from_pretrained(str(bert_dir))

tkn = tokenizer.tokenize('今日は、いい天気ですね。')
print(tkn)  # ['今日', '##は', '、', 'いい', '##天', '##気', '##で', '##す', '##ね', '。']
