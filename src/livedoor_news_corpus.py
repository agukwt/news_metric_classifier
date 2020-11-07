import pandas as pd

from src.util import CORPUS_DIR_PATH


__CORPUS_DIR_PATH_CHILD_NN = 'text'
CORPUS_INTEGRATE_PATH = CORPUS_DIR_PATH.joinpath(__CORPUS_DIR_PATH_CHILD_NN)


def extract_main_txt(file_path) -> str:
    """[summ本文を取得する前処理関数を定義]

    Parameters
    ----------
    file_path : [Path]
        [description]

    Returns
    -------
    text :[str]
        [description]
    """
    with open(file_path, encoding="utf-8") as text_file:
        # 今回はタイトル行は外したいので、3要素目以降の本文のみ使用
        text = text_file.readlines()[3:]

        # 3要素目以降にも本文が入っている場合があるので、リストにして、後で結合させる
        text = [sentence.strip() for sentence in text]  # 空白文字(スペースやタブ、改行)の削除
        text = list(filter(lambda line: line != '', text))
        text = ''.join(text)
        text = text.translate(str.maketrans(
            {'\n': '', '\t': '', '\r': '', '\u3000': ''}))  # 改行やタブ、全角スペースを消す
        return text


class Livedoor_News_Corpus(object):
    def __init__(self):
        self.categories = [p.name for p in CORPUS_INTEGRATE_PATH.iterdir() if p.is_dir()]
        self.corpus = None
        self.dic_id2cat = dict(zip(list(range(len(self.categories))), self.categories))
        self.dic_cat2id = dict(zip(self.categories, list(range(len(self.categories)))))

        return None

    def constract_coupus(self):
        list_text = []
        list_label = []

        for c_name in self.categories:
            # カテゴリーファイルを収集
            text_files = [p for p in CORPUS_INTEGRATE_PATH.joinpath(c_name).glob("{}*.txt".format(c_name)) if p.is_file]

            # 前処理extract_main_txtを実施して本文を取得
            body = [extract_main_txt(text_file) for text_file in text_files]

            label = [c_name] * len(body)  # bodyの数文だけカテゴリー名のラベルのリストを作成

            list_text.extend(body)  # appendが要素を追加するのに対して、extendはリストごと追加する
            list_label.extend(label)

            # pandasのDataFrameにする
            self.corpus = pd.DataFrame({'text': list_text, 'label': list_label})

        return self.corpus
    
    def change_category_to_id(self):
        # DataFrameにカテゴリーindexの列を作成
        self.corpus["label_index"] = self.corpus["label"].map(self.dic_cat2id)

        # label列を消去し、text, indexの順番にする
        self.corpus = self.corpus.loc[:, ["text", "label_index"]]
        return self.corpus

    def shufle_coupus(self):
        # TODO  random_stateの値の外在化
        return self.corpus.sample(frac=1, random_state=123).reset_index(drop=True)
