from pathlib import Path
import torch


# このファイルから相対的にPathを固定する（ついてはこのutil.pyの位置は動かしてはいけない）
this_dir = Path(__file__)

# DIR_Name(NM)
__DATA_DIR_NM = 'data'
__PREPROCESSED_DIR_NM = 'preprocessed'

__SRC_DIR_NM = 'news_metric_cluster'
__pretrained_BERT_NM = 'BERT-base_mecab-ipadic-bpe-32k_whole-word-mask'
CORPUS_DIR_NM = 'ldcc-20140209.tar'

# Global Variable
WORK_DIR_PATH = this_dir.parent.parent
DATA_DIR_PATH = WORK_DIR_PATH.joinpath(__DATA_DIR_NM)
CORPUS_DIR_PATH = DATA_DIR_PATH.joinpath(CORPUS_DIR_NM)

PREPROCESSED_DIR_PATH = DATA_DIR_PATH.joinpath(__PREPROCESSED_DIR_NM)
SRC_DIR_PATH = WORK_DIR_PATH.joinpath(__SRC_DIR_NM)
PRETRAINED_BERT_DIR = DATA_DIR_PATH.joinpath(__pretrained_BERT_NM)


def check_own_hw_unit():
    # GPU 確認
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())

    return None
