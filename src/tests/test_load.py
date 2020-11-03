import pytest

from src.load import load_tohoku_unv_bert_infomation


@pytest.fixture()
def bert_tokenizer():
    tohoku_unv_bert_tokenizer = load_tohoku_unv_bert_infomation()['tokenizer']
    return tohoku_unv_bert_tokenizer


def test_token_convert(bert_tokenizer):
    tkn = bert_tokenizer.tokenize('今日は、いい天気ですね。')
    convert = ['今日', '##は', '、', 'いい', '##天', '##気', '##で', '##す', '##ね', '。']

    assert tkn == convert
