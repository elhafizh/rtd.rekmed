import pandas as pd
import numpy as np
import re
import os
import pickle
from typing import List, Tuple
from dataclasses import dataclass


def drop_unnecessary_samples(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unused columns
    df.drop(["FS_ANAMNESA", "FS_CATATAN_FISIK"], axis=1, inplace=True)
    # Drop null diagnosa
    df.dropna(subset=["FS_DIAGNOSA"], inplace=True)
    column_1_nulls = df.FS_TINDAKAN.isnull()
    column_2_nulls = df.FS_TERAPI.isnull()
    # drop if tindakan & terapi both nulls
    intersect_nulls = set(df[column_1_nulls].index.tolist()).intersection(
        set(df[column_2_nulls].index.tolist())
    )
    intersect_nulls = list(intersect_nulls)
    df.drop(intersect_nulls, inplace=True)
    df.reset_index(drop=True)
    return df


def remove_html_tag(df: pd.DataFrame) -> pd.DataFrame:
    tag_pattern = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"
    over_space = "\s\s+"
    column_1_nulls = df.FS_TINDAKAN.isnull()
    column_2_nulls = df.FS_TERAPI.isnull()
    for i, row in df.iterrows():
        if not column_1_nulls[i]:
            row[0] = re.sub(tag_pattern, "", row[0])
            row[0] = re.sub("&nbsp;", "", row[0])
            row[0] = re.sub("---&gt;", "", row[0])
            row[0] = re.sub("--&gt;", "", row[0])
            row[0] = re.sub(over_space, " ", row[0]).strip()
        if not column_2_nulls[i]:
            row[1] = re.sub(tag_pattern, "", row[1])
            row[1] = re.sub("&nbsp;", "", row[1])
            row[1] = re.sub("---&gt;", "", row[1])
            row[1] = re.sub("--&gt;", "", row[1])
            row[1] = re.sub(over_space, " ", row[1]).strip()
        row[2] = re.sub(tag_pattern, "", row[2])
        row[2] = re.sub("&nbsp;", "", row[2])
        row[2] = re.sub("---&gt;", "", row[2])
        row[2] = re.sub("--&gt;", "", row[2])
        row[2] = re.sub(over_space, " ", row[2]).strip()
    df.rename(columns={
        "FS_TINDAKAN": "tindakan",
        "FS_TERAPI": "terapi",
        "FS_DIAGNOSA": "diagnosa"
    }, inplace=True)
    return df


output_path = "static/output/"


def df_save_output(df: pd.DataFrame, filename: str, extension: str) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        if extension == "xlsx":
            df.to_excel(f"{output_path}{filename}", index=False)
        elif extension == "tsv":
            df.to_csv(f"{output_path}{filename}", sep="\t")
    else:
        if extension == "xlsx":
            df.to_excel(f"{output_path}{filename}", index=False)
        elif extension == "tsv":
            df.to_csv(f"{output_path}{filename}", sep="\t")


def save_py_obj(filename: str, fileobj) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(f"{output_path}/{filename}", 'wb') as pkl_file:
        pickle.dump(fileobj, pkl_file)


def load_py_obj(filename: str) -> None:
    with open(f"{output_path}/{filename}", 'rb') as pkl_file:
        return pickle.load(pkl_file)


def check_total_tokens(list_of_tokens: List[List[str]]) -> int:
    total = 0
    for token in list_of_tokens:
        total = total + len(token)
    return total


def drop_meaningless_tokens(df: pd.DataFrame,
                            l_tokens: List[List[str]]) -> \
        Tuple[pd.DataFrame, List[List[str]]]:
    l_meaningless_token = ['mg']
    detect_index = []
    new_l_tokens = []
    for i, _ in enumerate(l_tokens):
        for lm in l_meaningless_token:
            if len(_) == 1:
                if lm == _[0]:
                    detect_index.append(i)
                else:
                    new_l_tokens.append(_)
            elif len(_) > 5:
                detect_index.append(i)
            else:
                new_l_tokens.append(_)
    df.drop(detect_index, inplace=True)
    df.reset_index(drop=True)
    print(f"drop {len(detect_index)} samples")
    return df, new_l_tokens


@dataclass
class SamplingQAPair:
    """Class for generating question & answer pair from diagnosa"""
    tindakan : str
    terapi : str
    diagnosa : str

    def generate(self) -> pd.DataFrame:
        if isinstance(self.tindakan, str):
            q_tindakan, a_tindakan = self.generate_tindakan()
        if isinstance(self.terapi, str):
            q_terapi, a_terapi = self.generate_terapi()
        if isinstance(self.tindakan, str) and isinstance(self.terapi, str):
            return pd.DataFrame({
                'question' : q_tindakan + q_terapi,
                'answer' : a_tindakan + a_terapi,
                'diagnosa': [self.diagnosa]*10
            })
        if isinstance(self.tindakan, str):
            return pd.DataFrame({
                'question' : q_tindakan,
                'answer' : a_tindakan,
                'diagnosa': [self.diagnosa]*5
            })
        if isinstance(self.terapi, str):
            return pd.DataFrame({
                'question' : q_terapi,
                'answer' : a_terapi,
                'diagnosa': [self.diagnosa]*5
            })

    def generate_tindakan(self) -> Tuple[List[str], List[str]]:
        q_tindakan = [
            f"apa tindakan yang dilakukan pada penyakit {self.diagnosa}?",
            f"tindakan apa yang tepat untuk menangani penyakit {self.diagnosa}?",
            f"{self.diagnosa} dapat ditangani dengan ...",
            f"bagaimana tindakan yg dilakukan untuk menangani penyakit {self.diagnosa}?",
            f"bagaimana {self.diagnosa} dapat ditangani?"
        ]
        a_tindakan = [self.tindakan]*5
        return q_tindakan, a_tindakan

    def generate_terapi(self) -> Tuple[List[str], List[str]]:
        q_terapi = [
            f"apa terapi yang tepat untuk penyakit {self.diagnosa}?",
            f"terapi apa yg disarankan untuk penderita {self.diagnosa}?",
            f"bagaimana {self.diagnosa} dapat diobati?",
            f"{self.diagnosa} dapat diobati dengan ...",
            f"terapi apa yang diperlukan untuk penyakit {self.diagnosa}?"
        ]
        a_terapi = [self.terapi]*5
        return q_terapi, a_terapi
    
