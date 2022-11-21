import pandas as pd
import numpy as np
import re
import os
import pickle
from typing import List


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


def drop_meaningless_tokens(df: pd.DataFrame, l_tokens: List[List[str]]) -> pd.DataFrame:
    l_meaningless_token = ['mg']
    detect_index = []
    for i, _ in enumerate(l_tokens):
        for lm in l_meaningless_token:
            if len(_) == 1:
                if lm == _[0]:
                    detect_index.append(i)
    df.drop(detect_index, inplace=True)
    df.reset_index(drop=True)
    return f"drop {len(detect_index)} samples"
