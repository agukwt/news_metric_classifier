import sys

import torch
from sklearn.model_selection import train_test_split

from src.dataset import make_textdata_dict
from src.dataset import make_contrastive_dataflame
from src.dataset import ContrastiveDataSet
from src.util import COTEGORY_INTEGRATE_DIR_PATH
from src.text_transfomer import BertTransform


def main():
    ##################################################
    # 1 データセット作成　　　　　
    ##################################################
    # カテゴリーとカテゴリーファイルの辞書を作成 　
    textdata_dict = make_textdata_dict(COTEGORY_INTEGRATE_DIR_PATH)
    # 作成する１つの対照情報の個数を指定して、対照情報を作成 (contrastive_num*2*カテゴリー数の行数を作成)
    contrastive_df = make_contrastive_dataflame(textdata_dict, contrastive_num=500)
    # 対照情報に対する変換処理を規定
    transform = BertTransform()
    # 対照情報を分割(train:val:test= 6:2:2)
    twenty_rasion_num = int(500 * 2 * 9 / 5)
    train_eval_df, test_df = train_test_split(contrastive_df, test_size=twenty_rasion_num, random_state=0)
    train_df, val_df = train_test_split(train_eval_df, test_size=twenty_rasion_num, random_state=0)
    # torchのDataSet化
    train_datasetx = ContrastiveDataSet(train_df, transform, phase='train')
    val_datasetx = ContrastiveDataSet(train_df, transform, phase='test')
    test_datasetx = ContrastiveDataSet(test_df, transform, phase='test')

    ##################################################
    # 2 データローダー作成　　　　　
    ##################################################
    batch_size = 32
    train_dataloder = torch.utils.data.DataLoader(train_datasetx, batch_size=batch_size, shuffle=True)
    val_dataloder = torch.utils.data.DataLoader(val_datasetx, batch_size=batch_size, shuffle=False)
    test_dataloder = torch.utils.data.DataLoader(test_datasetx, batch_size=batch_size, shuffle=False)

    sys.exit('Finish')


if __name__ == "__main__":
    main()

# %%
