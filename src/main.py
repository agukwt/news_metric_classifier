import sys

from src.dataset import make_textdata_dict
from src.dataset import make_contrastive_dataflame
from src.dataset import ContrastiveDataSet
from src.util import COTEGORY_INTEGRATE_DIR_PATH
from src.text_transfomer import BertTransform


def main():
    # %%
    ##################################################
    # 1 データセット作成　　　　　
    ##################################################
    # カテゴリーとカテゴリーファイルの辞書を作成 　
    textdata_dict = make_textdata_dict(COTEGORY_INTEGRATE_DIR_PATH)
    # 作成する１つの対照情報の個数を指定して、対照情報を作成 (contrastive_num*2*カテゴリー数の行数を作成)
    contrastive_df = make_contrastive_dataflame(textdata_dict, contrastive_num=500)
    # 対照情報に対する変換処理を規定
    transform = BertTransform()
    # 対照情報を分割
    # TODO: split
    train_df = contrastive_df
    test_df = contrastive_df
    # torchのDataSet化
    train_datasetx = ContrastiveDataSet(train_df, transform, phase='train')
    test_datasetx = ContrastiveDataSet(test_df, transform, phase='test')
    
    sys.exit('Finish')


if __name__ == "__main__":
    main()

# %%
