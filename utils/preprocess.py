import pandas as pd
import numpy as np
import re
import os


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
    return df


def df_save_output(df: pd.DataFrame, filename: str) -> None:
    output_path = "static/output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        df.to_excel(f"{output_path}{filename}", index=False)
    else:
        df.to_excel(f"{output_path}{filename}", index=False)
