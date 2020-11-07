# %%
from transformers import BertModel, BertConfig, BertJapaneseTokenizer
import torchtext

from src.util import DATA_DIR_PATH


class Bert(object):
    def __init__(self):
        self.bert_nm
        self.bert_dir

        self.model
        self.config
        self.tokenizer

        self.max_tkn_size = 512

    def set_pretrainned_model(self):
        self.model = BertModel.from_pretrained(str(self.bert_dir))

        return None

    def set_pretrainned_config(self):
        self.model = BertConfig(str(self.bert_dir))

        return None

    def set_pretrainned_tokenizer(self):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(str(self.bert_dir))
      
        return None


class JPN_Tohoku_Univ_Bert(Bert):
    def __init__(self):
        self.bert_nm = 'BERT-base_mecab-ipadic-bpe-32k_whole-word-mask'
        self.bert_dir = DATA_DIR_PATH.joinpath(self.bert_nm)
        self.max_tkn_size = 512

        self.tokenizer_method = None

        self.TEXT = None
        self.LABEL = None

    def tokenizer_512(self, input_text):
        """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
        self.max_tkn_size = 512

        return self.tokenizer.encode(
            input_text, truncation=True, max_length=self.max_tkn_size, padding=True, return_tensors='pt')[0]

    def set_torch_field(self, tokenizer_method):
        # torchtextのfieldsを定義
        self.TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_method, use_vocab=False, lower=False,
                                         include_lengths=True, batch_first=True, fix_length=self.max_tkn_size, pad_token=0)
        # 注意：tokenize=tokenizer.encodeと、.encodeをつけます。padding[PAD]のindexが0なので、0を指定します。
        self.LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
        return None

    def get_torch_field(self):
        return self.TEXT, self.LABEL
