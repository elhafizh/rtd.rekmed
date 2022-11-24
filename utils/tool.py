import pandas as pd
import numpy as np
import re
import os
import pickle
from typing import List, Tuple
from dataclasses import dataclass
from . import indexing
import os,pickle,random

import torch
from torch.utils.data import random_split
from sentence_transformers.readers import InputExample

from datetime import datetime
from sentence_transformers import (
    SentenceTransformer,
    SentencesDataset,
    losses,
    LoggingHandler,
)
import logging
from sklearn.model_selection import train_test_split
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from transformers import AutoTokenizer, AutoModel
from datasets import Dataset

from collections import defaultdict, Counter
from torch.utils.data import DataLoader
import math


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
    df.reset_index(drop=True, inplace=True)
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
    l_meaningless_token = ['mg', 'nv', 'non']
    detect_index = []
    for i, _ in enumerate(l_tokens):
        for lm in l_meaningless_token:
            if len(_) == 1:
                if lm == _[0]:
                    detect_index.append(i)
            elif len(_) > 4:
                detect_index.append(i)
    detect_index = list(set(detect_index))
    detect_index = sorted(detect_index, reverse=True)
    for index in detect_index:
        if index < len(l_tokens):
            l_tokens.pop(index)
    df.drop(detect_index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"drop {len(detect_index)} samples")
    return df, l_tokens


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
    

def qa_generator(df: pd.DataFrame) -> pd.DataFrame:
    rekmed_qa = pd.DataFrame()
    for i, row in df.iterrows():
        sampling_qa_pair = SamplingQAPair(
            row[0],row[1],row[2]
        )
        rekmed_qa = pd.concat([rekmed_qa, sampling_qa_pair.generate()])
    rekmed_qa.reset_index(drop=True, inplace=True)
    return rekmed_qa


def grouping_qa(df: pd.DataFrame) -> pd.DataFrame:
    l_tokens = indexing.tokenization(df.diagnosa)
    df, l_tokens = drop_meaningless_tokens(df, l_tokens)
    group = []
    for token in l_tokens:
        group.append(f"g{token[0][:4]}")
    new_df = pd.DataFrame({
        'group' : group,
        'question' : df.question,
        'answer' : df.answer
    })
    return new_df


def gather_entailment(df: pd.DataFrame) -> pd.DataFrame:
    group1: List[str] = []
    group2: List[str] = []
    question1: List[str] = []
    question2: List[str] = []
    label: List[float] = []
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j:
                if (row_i.group == row_j.group) and (row_i.answer == row_j.answer):
                    # prevent duplication
                    if row_j.question not in question1 and row_i.question not in question2:
                        group1.append(row_i.group)
                        group2.append(row_j.group)
                        question1.append(row_i.question)
                        question2.append(row_j.question)
                        label.append(0.8)
    return pd.DataFrame({
        'group1' : group1,
        'group2' : group2,
        'question1' : question1,
        'question2' : question2,
        'label' : label
    })


def gather_neutral(df: pd.DataFrame) -> pd.DataFrame:
    group1: List[str] = []
    group2: List[str] = []
    question1: List[str] = []
    question2: List[str] = []
    label: List[float] = []
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j:
                if (row_i.group == row_j.group) and (row_i.answer != row_j.answer):
                    # prevent duplication
                    if row_j.question not in question1 and row_i.question not in question2:
                        group1.append(row_i.group)
                        group2.append(row_j.group)
                        question1.append(row_i.question)
                        question2.append(row_j.question)
                        label.append(0.4)
    return pd.DataFrame({
        'group1' : group1,
        'group2' : group2,
        'question1' : question1,
        'question2' : question2,
        'label' : label
    })


def gather_contradiction(df: pd.DataFrame) -> pd.DataFrame:
    group1: List[str] = []
    group2: List[str] = []
    question1: List[str] = []
    question2: List[str] = []
    label: List[float] = []
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j:
                if (row_i.group != row_j.group) and (row_i.answer != row_j.answer):
                    # prevent duplication
                    if row_j.question not in question1 and row_i.question not in question2:
                        group1.append(row_i.group)
                        group2.append(row_j.group)
                        question1.append(row_i.question)
                        question2.append(row_j.question)
                        label.append(0)
    return pd.DataFrame({
        'group1' : group1,
        'group2' : group2,
        'question1' : question1,
        'question2' : question2,
        'label' : label
    })


def concatenate_train_examples(l_entailment: pd.DataFrame, l_neutral: pd.DataFrame,
                          l_contradict: pd.DataFrame) -> pd.DataFrame:
    num = len(l_entailment)
    downsize = 0
    for i in [10000,1000,100,10]:
        if num//i:
            downsize = i*10 # if i==1000, downsize dataset by 1000*10
            break
    frames = [l_entailment, l_neutral.sample(n=downsize), l_contradict.sample(n=downsize)]
    print(len(l_neutral.sample(n=downsize)))
    print(len(l_contradict.sample(n=downsize)))
    result = pd.concat(frames).reset_index(drop=True).loc[:, "question1":"label"]
    return result


class STSModel:
    def __init__(
        self,
        pretrained_model_name,
        cuda_device="cuda",
        pooling_option="mean",
        task="sts",
        local=False
    ):

        self.pretrained_model_name = pretrained_model_name
        self.cuda_device = cuda_device
        if not os.path.exists("static/output/models"):
            os.makedirs("static/output/models")
        if not local:
            self.model_output = f"static/output/models/{pretrained_model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{pooling_option}-{task}"
        else:
            self.model_output = self.pretrained_model_name

        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[LoggingHandler()],
        )
        self.model = SentenceTransformer(self.pretrained_model_name)
        self.device = torch.device(self.cuda_device)
        self.model.to(self.device)

    def get_samples(self, df):
        """Convert dataset into InputExample of Sentence Transformer"""
        samples = []
        for i, row in df.iterrows():
            inp_example = InputExample(texts=[row.question1, row.question2], label=row.label)
            samples.append(inp_example)
        return samples

    def get_train_samples(self, df):
        self.train_samples = self.get_samples(df)

    def get_dev_samples(self, df):
        self.dev_samples = self.get_samples(df)

    def get_test_samples(self, df):
        # self.test_samples = self.get_samples(df)
        self.test_samples = self.get_samples(df)

    def get_train_test_dataset(self, df):
        """Separate samples into 90% Training & 10% Testing"""
        X = df.loc[:, df.columns[0:2]]
        y = df.loc[:, df.columns[2]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9, random_state=42, shuffle=True
        )
        self.get_train_samples(
            pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        )
        self.get_dev_samples(
            pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
        )

    def fit(self, n_batch=16, n_epoch=4):
        logging.info("Read Train Dataset")
        train_dataloader = DataLoader(
            self.train_samples, shuffle=True, batch_size=n_batch
        )
        logging.info("Read Validation Dataset")
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            self.dev_samples, name="sts-dev"
        )

        warmup_steps = math.ceil(
            len(self.train_samples) * n_epoch / n_batch * 0.1
        )  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=n_epoch,
            evaluation_steps=int(len(train_dataloader) * 0.1),
            warmup_steps=warmup_steps,
            output_path=self.model_output,
        )

    def test(self):
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            self.test_samples, name="sts-test"
        )
        similarity = test_evaluator(self.model)
        logging.info(f"Test {self.model_output}")
        logging.info(f"similarity: {similarity}")
        return similarity


class LoadFTModel:
    def __init__(self, pre_trained_model, cuda_device="cuda"):
        self.pre_trained_model = pre_trained_model
        self.cuda_device = cuda_device

    def proceed(self):
        model_ckpt = self.pre_trained_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt)

        self.device = torch.device(self.cuda_device)
        self.model.to(self.device)
        return self.model

    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)

