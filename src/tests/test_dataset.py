import pytest
from pathlib import Path

from src.dataset import make_textdata_dict
from src.dataset import make_complement_file_list
from src.dataset import make_positive_categary_pairs
from src.dataset import make_negative_categary_pairs
from src.dataset import make_contrastive_posinegadata_on_category
from src.dataset import make_contrastive_dataflame_on_category
from src.dataset import make_contrastive_dataflame
from src.util import CORPUS_DIR_PATH

__CORPUS_DIR_PATH_CHILD_NN = 'text'
CORPUS_INTEGRATE_PATH = CORPUS_DIR_PATH.joinpath(__CORPUS_DIR_PATH_CHILD_NN)


@pytest.fixture
def text_dict():
    # 前処理
    text_dict = make_textdata_dict(CORPUS_INTEGRATE_PATH)

    yield text_dict   # テスト関数に何らかの値を渡す

    # 後処理


def test_get_category(text_dict):
    assert "dokujo-tsushin" == list(text_dict.keys())[0]


def test_make_textdata_dict_path(text_dict):
    check_head_path = text_dict[list(text_dict.keys())[0]][0]
    check_tail_path = text_dict[list(text_dict.keys())[-1]][-1]
    
    # head
    assert CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-4778030.txt") == Path(check_head_path)
    
    # tail
    assert CORPUS_INTEGRATE_PATH.joinpath("topic-news", "topic-news-6918105.txt") == Path(check_tail_path)


def test_complement_filelist(text_dict):
    comp_list = make_complement_file_list(text_dict, "dokujo-tsushin")
    assert CORPUS_INTEGRATE_PATH.joinpath("it-life-hack", "it-life-hack-6292880.txt") == Path(comp_list[0])
    assert CORPUS_INTEGRATE_PATH.joinpath("topic-news", "topic-news-6918105.txt") == Path(comp_list[-1])


def test_positive_pair(text_dict):
    positive_pairs = make_positive_categary_pairs(text_dict, "dokujo-tsushin")

    # positive_pairsの構造:= [(first_antecedent, first_below), ... ,(last_antecedent, last_below)]
    first_antecedent = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-4778030.txt")
    first_descendant = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-4778031.txt")
    last_antecedent = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-6910523.txt")
    last_descendant = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-6915005.txt")

    assert first_antecedent == Path(positive_pairs[0][0]) and first_descendant == Path(positive_pairs[0][1])
    assert last_antecedent == Path(positive_pairs[-1][0]) and last_descendant == Path(positive_pairs[-1][1])


def test_negative_pair(text_dict):
    negative_pairs = make_negative_categary_pairs(text_dict, "dokujo-tsushin")

    # negative_pairsの構造:= [(pos_first, neg_first),(pos_first, neg_second), ... ,(pos_last, neg_last)]
    pos_first = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-4778030.txt")
    neg_first = CORPUS_INTEGRATE_PATH.joinpath("it-life-hack", "it-life-hack-6292880.txt")
    pos_last = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-6915005.txt")
    neg_last = CORPUS_INTEGRATE_PATH.joinpath("topic-news", "topic-news-6918105.txt")

    assert pos_first == Path(negative_pairs[0][0]) and neg_first == Path(negative_pairs[0][1])
    assert pos_last == Path(negative_pairs[-1][0]) and neg_last == Path(negative_pairs[-1][1])


def test_make_contrastive_posinegadata_on_category(text_dict):
    category = "dokujo-tsushin"

    # 対照学習のデータ作成数の確認
    num = 200
    positive_pairs, negative_pairs = make_contrastive_posinegadata_on_category(text_dict, category, num)
    assert num == len(positive_pairs)
    assert num == len(negative_pairs)
    
    # 再現性の確認
    _positive_pairs, _negative_pairs = make_contrastive_posinegadata_on_category(text_dict, category, num)
    assert positive_pairs[0] == _positive_pairs[0]
    assert positive_pairs[-1] == _positive_pairs[-1]
    assert negative_pairs[0] == _negative_pairs[0]
    assert negative_pairs[-1] == _negative_pairs[-1]


def test_make_contrastive_dataflame_on_category(text_dict):
    category = "dokujo-tsushin"

    num = 100
    positive_pairs, negative_pairs = make_contrastive_posinegadata_on_category(text_dict, category, num)
    contrastive_df = make_contrastive_dataflame_on_category(category, positive_pairs, negative_pairs)

    first_antecedent = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-5598425.txt")
    first_descendant = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-6076462.txt")
    last_antecedent = CORPUS_INTEGRATE_PATH.joinpath("dokujo-tsushin", "dokujo-tsushin-6765514.txt")
    last_descendant = CORPUS_INTEGRATE_PATH.joinpath("sports-watch", "sports-watch-5710714.txt")

    assert first_antecedent == Path(contrastive_df.iloc[0]['antecedent_path'])
    assert first_descendant == Path(contrastive_df.iloc[0]['descendant_path'])
    assert last_antecedent == Path(contrastive_df.iloc[-1]['antecedent_path'])
    assert last_descendant == Path(contrastive_df.iloc[-1]['descendant_path'])


def test_make_contrastive_dataflame(text_dict):
    contrastive_df = make_contrastive_dataflame(text_dict, 50)

    assert ['category', 'contrastive_viewpoint', 'antecedent_path', 'descendant_path'] == contrastive_df.columns.to_list()
    assert (50 * 2 * 9, 4) == contrastive_df.shape
