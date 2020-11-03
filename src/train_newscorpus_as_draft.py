# %% import
# import os
# import pickle
# import pprint
import glob
import pandas as pd
import torchtext
import random
import torch
from torch import nn
import torch.optim as optim

from src.util import PREPROCESSED_DIR_PATH
# from src.util import check_own_hw_unit
from src.load import load_tohoku_unv_bert_infomation
from src.util import DATA_DIR_PATH, CORPUS_DIR_NM


# %%
###################################################################
# 1. livedoorニュースをダウンロードして、tsvファイルに変換           #
###################################################################
# コーパス：ivedoorニュースのロード
# 本文を取得する前処理関数を定義
def extract_main_txt(file_name):
    with open(file_name, encoding="utf-8") as text_file:
        # 今回はタイトル行は外したいので、3要素目以降の本文のみ使用
        text = text_file.readlines()[3:]

        # 3要素目以降にも本文が入っている場合があるので、リストにして、後で結合させる
        text = [sentence.strip() for sentence in text]  # 空白文字(スペースやタブ、改行)の削除
        text = list(filter(lambda line: line != '', text))
        text = ''.join(text)
        text = text.translate(str.maketrans(
            {'\n': '', '\t': '', '\r': '', '\u3000': ''}))  # 改行やタブ、全角スペースを消す
        return text


# %%
# リストに前処理した本文と、カテゴリーのラベルを追加していく
category = {
    "dokujo-tsushin": 1,
    "it-life-hack": 2,
    "kaden-channel": 3,
    "livedoor-homme": 4,
    "movie-enter": 5,
    "peachy": 6,
    "smax": 7,
    "sports-watch": 8,
    "topic-news": 9
}

list_text = []
list_label = []

# ファイルのパスを取得
ldcc_dir = DATA_DIR_PATH.joinpath(CORPUS_DIR_NM)

for c_name, c_id in category.items():
    # カテゴリーファイルを収集
    text_files = glob.glob(str(ldcc_dir) + "/text/{c_name}/{c_name}*.txt".format(c_name=c_name))

    # 前処理extract_main_txtを実施して本文を取得
    body = [extract_main_txt(text_file) for text_file in text_files]

    label = [c_name] * len(body)  # bodyの数文だけカテゴリー名のラベルのリストを作成

    list_text.extend(body)  # appendが要素を追加するのに対して、extendはリストごと追加する
    list_label.extend(label)
# %%
# 0番目の文章とラベルを確認
print(list_text[0][:50])
print(list_label[0])

# %%
# pandasのDataFrameにする
df = pd.DataFrame({'text': list_text, 'label': list_label})

# 大きさを確認しておく（7,376文章が存在）
print(df.shape)

# カテゴリーの辞書を作成
categories = ['dokujo-tsushin', 'it-life-hack', 'smax', 'sports-watch', 'kaden-channel', 'movie-enter', 'topic-news', 'livedoor-homme', 'peachy']
dic_id2cat = dict(zip(list(range(len(categories))), categories))
dic_cat2id = dict(zip(categories, list(range(len(categories)))))

print(dic_id2cat)
print(dic_cat2id)

# DataFrameにカテゴリーindexの列を作成
df["label_index"] = df["label"].map(dic_cat2id)
df.head()

# label列を消去し、text, indexの順番にする
df = df.loc[:, ["text", "label_index"]]
df.head()

# 順番をシャッフルする
df = df.sample(frac=1, random_state=123).reset_index(drop=True)
df.head()

# %%
# tsvファイルで保存する
len_0_2 = len(df) // 5

# 前から2割をテストデータとする
df[:len_0_2].to_csv(PREPROCESSED_DIR_PATH.joinpath("test.tsv"), sep='\t', index=False, header=None)
print(df[:len_0_2].shape)

# 前2割からを訓練&検証データとする
df[len_0_2:].to_csv(PREPROCESSED_DIR_PATH.joinpath("train_eval.tsv"), sep='\t', index=False, header=None)
print(df[len_0_2:].shape)

# %%
###################################################################
# 2. tsvファイルをPyTorchのtorchtextのDataLoaderに変換             #
###################################################################
pt_bert = load_tohoku_unv_bert_infomation()
model = pt_bert['model']
config = pt_bert['config']
tokenizer = pt_bert['tokenizer']

max_length = 512  # 東北大学_日本語版の最大の単語数（サブワード数）は512


def tokenizer_512(input_text):
    """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
    return tokenizer.encode(input_text, truncation=True, max_length=512, pad_to_max_length=True, return_tensors='pt')[0]


TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_512, use_vocab=False, lower=False,
                            include_lengths=True, batch_first=True, fix_length=max_length, pad_token=0)
# 注意：tokenize=tokenizer.encodeと、.encodeをつけます。padding[PAD]のindexが0なので、0を指定します。

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

# (注釈)：各引数を再確認
# sequential: データの長さが可変か？文章は長さがいろいろなのでTrue.ラベルはFalse
# tokenize: 文章を読み込んだときに、前処理や単語分割をするための関数を定義
# use_vocab：単語をボキャブラリーに追加するかどうか
# lower：アルファベットがあったときに小文字に変換するかどうか
# include_length: 文章の単語数のデータを保持するか
# batch_first：ミニバッチの次元を用意するかどうか
# fix_length：全部の文章をfix_lengthと同じ長さになるように、paddingします
# init_token, eos_token, pad_token, unk_token：文頭、文末、padding、未知語に対して、どんな単語を与えるかを指定

dataset_train_eval, dataset_test = torchtext.data.TabularDataset.splits(
    path=PREPROCESSED_DIR_PATH, train='train_eval.tsv', test='test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

# %%
dataset_train, dataset_eval = dataset_train_eval.split(
    split_ratio=1.0 - 1473 / 5894, random_state=random.seed(1234))

# datasetの長さを確認してみる
print(dataset_train.__len__())
print(dataset_eval.__len__())
print(dataset_test.__len__())

# %%
# datasetの中身を確認してみる
item = next(iter(dataset_train))
print(item.Text)
print("長さ：", len(item.Text))  # 長さを確認 [CLS]から始まり[SEP]で終わる。512より長いと後ろが切れる
print("ラベル：", item.Label)
# %%
# datasetの中身を文章に戻し、確認
print(tokenizer.convert_ids_to_tokens(item.Text.tolist()))  # 文章
print(dic_id2cat[int(item.Label)])  # id

# %%
# DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
batch_size = 32  # BERTでは16、32あたりを使用する

dl_train = torchtext.data.Iterator(
    dataset_train, batch_size=batch_size, train=True)

dl_eval = torchtext.data.Iterator(
    dataset_eval, batch_size=batch_size, train=False, sort=False)

dl_test = torchtext.data.Iterator(
    dataset_test, batch_size=batch_size, train=False, sort=False)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": dl_train, "val": dl_eval}
# DataLoaderの動作確認
batch = next(iter(dl_test))
print(batch)
print(batch.Text[0].shape)
print(batch.Label.shape)

# %%
###################################################################
# 3. BERTのクラス分類用のモデルを用意する                           #
###################################################################
# BERTの日本語学習済みパラメータのモデルです
print(model)


# %%
class BertForLivedoor(nn.Module):
    '''BERTモデルにLivedoorニュースの9クラスを判定する部分をつなげたモデル'''

    def __init__(self):
        super(BertForLivedoor, self).__init__()

        # BERTモジュール
        self.bert = model  # 日本語学習済みのBERTモデル

        # headにクラス予測を追加
        # 入力はBERTの出力特徴量の次元768、出力は9クラス
        self.cls = nn.Linear(in_features=768, out_features=9)

        # 重み初期化処理
        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)

    def forward(self, input_ids):
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        '''

        # BERTの基本モデル部分の順伝搬
        # 順伝搬させる
        result = self.bert(input_ids)  # reult は、sequence_output, pooled_output

        # sequence_outputの先頭の単語ベクトルを抜き出す
        vec_0 = result[0]  # 最初の0がsequence_outputを示す
        vec_0 = vec_0[:, 0, :]  # 全バッチ。先頭0番目の単語の全768要素
        vec_0 = vec_0.view(-1, 768)  # sizeを[batch_size, hidden_size]に変換
        output = self.cls(vec_0)  # 全結合層

        return output


# %%
# モデル構築
net = BertForLivedoor()

# 訓練モードに設定
net.train()

print('ネットワーク設定完了')

# %%
###################################################################
# 4. ファインチューニングの設定                                     #
###################################################################
# 勾配計算を最後のBertLayerモジュールと追加した分類アダプターのみ実行

# 1. まず全部を、勾配計算Falseにしてしまう
for param in net.parameters():
    param.requires_grad = False

# 2. BertLayerモジュールの最後を勾配計算ありに変更
for param in net.bert.encoder.layer[-1].parameters():
    param.requires_grad = True

# 3. 識別器を勾配計算ありに変更
for param in net.cls.parameters():
    param.requires_grad = True

# 最適化手法の設定
# BERTの元の部分はファインチューニング
optimizer = optim.Adam([
    {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
    {'params': net.cls.parameters(), 'lr': 1e-4}
])

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likelihood loss)を計算


# %%
###################################################################
# 5. 学習の実施                                                   #
###################################################################
# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    # ネットワークをGPUへ
    # net.to(device)
    net.cuda()  # GPU対応

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = dataloaders_dict["train"].batch_size

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            iteration = 1

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書型変数
                
                # GPUが使えるならGPUにデータを送る
                # inputs = batch.Text[0].to(device)  # 文章
                # labels = batch.Label.to(device)  # ラベル
                inputs = batch.Text[0].cuda()  # 文章
                labels = batch.Label.cuda()  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # BERTに入力
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)  # 損失を計算

                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            acc = (torch.sum(preds == labels.data)
                                   ).double() / batch_size
                            print('イテレーション {} || Loss: {:.4f} || 10iter. || 本イテレーションの正解率：{}'.format(
                                iteration, loss.item(), acc))

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))

    return net


# 学習・検証を実行する。1epochに2分ほどかかります
num_epochs = 10
net_trained = train_model(
    net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
# %%
