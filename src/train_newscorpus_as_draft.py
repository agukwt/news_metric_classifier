import sys

from src.dump import dump_to_pickle
from src.load import load_from_pickle
from src.util import DATA_DIR_PATH
from src.livedoor_news_corpus import Livedoor_News_Corpus


###################################################################
# 0. 実行記録の切り替えための変数定義                                #
###################################################################
folder_nm = 'dev01'
output_folder_path = DATA_DIR_PATH.joinpath(folder_nm)

###################################################################
# 1. livedoorニュースデータの用意                                  #
###################################################################
corpus_path = output_folder_path.joinpath("corpus.pickle")

# corpus作成・保存
cps = Livedoor_News_Corpus()  # インスタンス作成
corpus_df = cps.constract_coupus()  # コーパスデータ構築
corpus_df = cps.change_category_to_id()  # カテゴリー値からID値に置き換え
corpus_df = cps.shufle_coupus()  # コーパスデータのシャッフル
dump_to_pickle(corpus_path, corpus_df)
del corpus_df

# corpus読み込み
corpus_df = load_from_pickle(corpus_path)

sys.exit("")
