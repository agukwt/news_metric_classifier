import pytest

from src.bert import JPN_Tohoku_Univ_Bert


@pytest.fixture()
def japanese_bert():
    jpn_bert = JPN_Tohoku_Univ_Bert()

    return jpn_bert


def test_set_pretrainned_model(japanese_bert):
    japanese_bert.set_pretrainned_model()

    assert japanese_bert.model is not None


def test_set_pretrainned_config(japanese_bert):
    japanese_bert.set_pretrainned_config()

    assert japanese_bert.config is not None


def test_set_pretrainned_tokenizer(japanese_bert):
    japanese_bert.set_pretrainned_tokenizer()

    assert japanese_bert.tokenizer is not None


def test_token_convert(japanese_bert):
    japanese_bert.set_pretrainned_tokenizer()
    tkn = japanese_bert.tokenizer.tokenize('今日は、いい天気ですね。')
    convert = ['今日', '##は', '、', 'いい', '##天', '##気', '##で', '##す', '##ね', '。']

    assert tkn == convert
