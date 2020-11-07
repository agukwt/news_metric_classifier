import random

import torchtext
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split

from src.model import check_GPU
from src.model import BertForLivedoor
from src.model import train_model
from src.model import fintune_bert_model

from src.bert import JPN_Tohoku_Univ_Bert
from src.dump import dump_to_pickle
from src.load import load_from_pickle
from src.util import DATA_DIR_PATH, TRAIN_EVAL_NM, TEST_NM
from src.livedoor_news_corpus import Livedoor_News_Corpus


###################################################################
# 0. 実行記録の切り替えための変数定義                                #
###################################################################
folder_nm = 'dev01'
output_folder_path = DATA_DIR_PATH.joinpath(folder_nm)
check_GPU()

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


###################################################################
# 2. コーパスのtorchtextを利用したDataLoaderへの変換                 #
###################################################################
# トークンナイザーのエンコーダーを設定する
bert = JPN_Tohoku_Univ_Bert()
bert.set_pretrainned_tokenizer()
# トークンナイザー方法を指定して、torchtextのfields作成と呼び出し
bert.set_torch_field(bert.tokenizer_512)
TEXT, LABEL = bert.get_torch_field()

# datasetのtrain_eval(train, eval), testの分割
# train_evalとtestの分割
train_eval_df, test_df = train_test_split(
    corpus_df, test_size=0.2, shuffle=True, random_state=random.seed(1234), stratify=corpus_df[['label_index']])
# tsv保存
train_eval_df.to_csv(output_folder_path.joinpath(TRAIN_EVAL_NM), sep='\t', index=False, header=None)
test_df.to_csv(output_folder_path.joinpath(TEST_NM), sep='\t', index=False, header=None)

# torchtext化
dataset_train_eval, dataset_test = torchtext.data.TabularDataset.splits(
    path=output_folder_path, train=TRAIN_EVAL_NM, test=TEST_NM, format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

# trainとevalの分割
ratio = 1.0 - (len(dataset_test) / len(dataset_train_eval))
dataset_train, dataset_eval = dataset_train_eval.split(
    split_ratio=ratio, random_state=random.seed(1234))

# DataLoader作成（torchtextの文脈では単純にiteraterと呼ばれる）
batch_size = 32  # BERTでは16、32あたりを使用
dl_train = torchtext.data.Iterator(dataset_train, batch_size=batch_size, train=True)
dl_eval = torchtext.data.Iterator(dataset_eval, batch_size=batch_size, train=False, sort=False)
dl_test = torchtext.data.Iterator(dataset_test, batch_size=batch_size, train=False, sort=False)
# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": dl_train, "val": dl_eval}


###################################################################
# 3. BERTのクラス分類用のモデルを用意する                           #
###################################################################
# BERTの日本語学習済みパラメータのモデルです
bert.set_pretrainned_model()

# モデル構築
net = BertForLivedoor(base_bart=bert.model)
# 訓練モードに設定
net.train()


###################################################################
# 4. ファインチューニングの設定                                     #
###################################################################
# 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行
# 1. まず全部を、勾配計算Falseにしてしまう
fintune_bert_model(net)

# 最適化手法の設定(# BERTの元の部分はファインチューニング)
optimizer = optim.Adam([
    {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': net.cls.parameters(), 'lr': 1e-4}
])

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算


###################################################################
# 5. 学習の実施                                                   #
###################################################################
# 学習・検証を実行する。1epochに2分ほどかかります
num_epochs = 1
net_trained = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
