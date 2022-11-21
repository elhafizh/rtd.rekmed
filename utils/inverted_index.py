import numpy as np
import pandas as pd
import string
import pickle
import spacy
import time
from tabulate import tabulate
from tqdm.notebook import tqdm
from tqdm import tqdm
import nltk
import re
from nltk.tokenize import RegexpTokenizer
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from typing import List, Sequence


# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
# nltk.download('punkt')
nlp_id = spacy.blank('id')


def tokenization(contents: pd.Series) -> List[List[str]]:
    list_tokens_from_docs = []
    for content in tqdm(contents):
        # sentence stemming
        # content = stemmer.stem(content)
        nlp_contents = nlp_id(content)
        clean_token = []
        for token_of_nlp_contents in nlp_contents:
            # remove punctuation & remove stopword
            if not token_of_nlp_contents.is_digit and not token_of_nlp_contents.is_punct \
                    and not token_of_nlp_contents.is_stop:
                clean_token.append(token_of_nlp_contents.text.lower())
        list_tokens_from_docs.append(clean_token)
    return list_tokens_from_docs


def remove_alphanumeric_char(nlp_tokens: List[List[str]]):
    # example : --sebagai, -sebagai
    hyphen_separator = RegexpTokenizer(r'\w+')
    # example : 000anak, 000golongan
    num_letter_separator = RegexpTokenizer(r'\d+|\D+')
    clean_tokens = []
    for tokens in tqdm(nlp_tokens):
        clean_token_ofnum = []
        for token in tokens:
            pre_tokens = num_letter_separator.tokenize(token)
            for pre_token in pre_tokens:
                if len(pre_token) > 1:
                    pre_token = nlp_id(pre_token)
                    if not pre_token[0].is_digit and not pre_token[0].is_punct:
                        clean_token_ofnum.append(pre_token[0].text)
        clean_token_ofhyp = []
        for token in clean_token_ofnum:
            pre_tokens = hyphen_separator.tokenize(token)
            clean_token_ofhyp.append(pre_tokens[0])
        clean_tokens.append(clean_token_ofhyp)
    return clean_tokens

if __name__ == "__main__":
    pass
