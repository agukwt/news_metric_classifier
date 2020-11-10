import itertools
import random
import pandas as pd
from torch.utils import data


from src.util import CORPUS_DIR_PATH


__CORPUS_DIR_PATH_CHILD_NN = 'text'
CORPUS_INTEGRATE_PATH = CORPUS_DIR_PATH.joinpath(__CORPUS_DIR_PATH_CHILD_NN)


def make_textdata_dict(text_dir):
    """ パスから、子ディレクトリにあるカテゴリーに基づくテキストファイルリストを持つ、辞書を作る

    Parameters
    ----------
    text_dir : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # text_dir以下にある、子フォルダ名をcategoriesとする
    categories = [p.name for p in text_dir.iterdir() if p.is_dir()]

    # カテゴリーをkeyとして、テキストパスリストを作る
    textfile_dict = {}
    for c_name in categories:
        # カテゴリーファイルを収集
        text_files = [str(p) for p in text_dir.joinpath(c_name).glob("{}*.txt".format(c_name)) if p.is_file]
        textfile_dict[c_name] = text_files
    
    return textfile_dict


def make_complement_file_list(textfile_dict: dict, category: str) -> list:
    """ textfile_dictで指定させたcategoryでないファイルリストを作成する

    Parameters
    ----------
    textfile_dict : dict
        [description]
    category : str
        [description]

    Returns
    -------
    list
        [description]
    """

    complement_list = []

    for k in textfile_dict.keys():
        if k == category:
            pass
        else:
            complement_list.extend(textfile_dict[k])
    
    return complement_list


def make_positive_categary_pairs(textfile_dict: dict, category: str) -> list:
    categary_list = textfile_dict[category]
    categary_pairs_list = [p for p in itertools.combinations(categary_list, 2)]

    return categary_pairs_list


def make_negative_categary_pairs(textfile_dict: dict, category: str) -> list:
    base_list = textfile_dict[category]
    complement_list = make_complement_file_list(textfile_dict, category)
    categary_pairs_list = [p for p in itertools.product(base_list, complement_list)]

    return categary_pairs_list


def make_contrastive_posinegadata_on_category(textfile_dict: dict, category: str, contrastive_num=100):
    positive_pairs = make_positive_categary_pairs(textfile_dict, category)
    negative_pairs = make_negative_categary_pairs(textfile_dict, category)
    
    limit_num = contrastive_num

    positive_num = len(positive_pairs)
    negtive_num = len(negative_pairs)
    
    if negtive_num < positive_num:
        if negtive_num < contrastive_num:
            limit_num = negtive_num
        else:
            pass
    else:
        if positive_num < contrastive_num:
            limit_num < positive_num
        else:
            pass
    
    random.seed(1234)
    
    positive_limit_pairs = random.sample(positive_pairs, limit_num)
    negative_limit_pairs = random.sample(negative_pairs, limit_num)

    return positive_limit_pairs, negative_limit_pairs


def make_contrastive_dataflame_on_category(category, positive_pairs, negative_pairs):
    cols = ['category', 'contrastive_viewpoint', 'antecedent_path', 'descendant_path']
    contrastive_df = pd.DataFrame(index=[], columns=cols)

    p_df = pd.DataFrame(positive_pairs, columns=[cols[2], cols[3]])
    p_df[cols[1]] = 'positive'
    p_df[cols[0]] = category

    n_df = pd.DataFrame(negative_pairs, columns=[cols[2], cols[3]])
    n_df[cols[1]] = 'negative'
    n_df[cols[0]] = category

    contrastive_df = pd.concat([contrastive_df, p_df], axis=0)
    contrastive_df = pd.concat([contrastive_df, n_df], axis=0)

    return contrastive_df.reset_index(drop=True)


def make_contrastive_dataflame(textfile_dict: dict, contrastive_num=250):
    cols = ['category', 'contrastive_viewpoint', 'antecedent_path', 'descendant_path']
    contrastive_df = pd.DataFrame(index=[], columns=cols)
    
    for category in textfile_dict.keys():
        positive_pairs, negative_pairs = make_contrastive_posinegadata_on_category(textfile_dict, category, contrastive_num)
        cat_contrastive_df = make_contrastive_dataflame_on_category(category, positive_pairs, negative_pairs)
        contrastive_df = pd.concat([contrastive_df, cat_contrastive_df], axis=0)
    
    return contrastive_df.reset_index(drop=True)


class ContrastiveDataSet(data.Dataset):
    def __init__(self, constractive_data, transform, phase):
        self.constractive_data = constractive_data
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return self.constractive_data.shape[0]
    
    def __getitem__(self, index):
        """ transformed text と label(positive:0 /negative:1 )を取得
        """

        # index番目のデータフレームから、transformed text と labelを作成
        # index番目のデータフレーム選択
        cont_object = self.constractive_data.iloc[index]
        # 定義による変換結果を取得
        transformed_antecedent = self.transform(cont_object['antecedent_path'], self.phase)
        transformed_descendant = self.transform(cont_object['descendant_path'], self.phase)
        # 対照観点が'positive'なら、0を、そうでなければ、1を代入
        label = 0 if cont_object['contrastive_viewpoint'] == 'positive' else 1

        return transformed_antecedent, transformed_descendant, label
