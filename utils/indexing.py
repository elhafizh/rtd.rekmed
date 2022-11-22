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
from collections import deque
from nltk.tokenize import RegexpTokenizer
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from typing import List, Sequence, Tuple, Dict, Deque


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


def pair_token_docID(nlp_tokens: List[List[str]]) -> List[Tuple[str, int]]:
    doc_list_ofToken_DocID = []
    # index_docID
    for docID, doc_list_T in tqdm(enumerate(nlp_tokens)):
        for doc_list in doc_list_T:
            # doc_list_ofToken_DocID.append((doc_list.text, docID+1))
            doc_list_ofToken_DocID.append((doc_list, docID+1))
    return doc_list_ofToken_DocID


def remove_space(nlp_tokens_tup: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    new_tokens = []
    for i in nlp_tokens_tup:
        if not i[0].isspace():
            new_tokens.append(i)
    return new_tokens


def sort_token_docID(tup: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    return (sorted(tup, key=lambda x: x[0]))


def inverted_index(tokens_docID: List[Tuple[str, int]]) -> \
        Dict[Tuple[str, int], Deque[int]]:
    vocab = {}
    for token, docId in tqdm(tokens_docID):
        if not token in vocab:
            vocab[token] = deque()
            vocab[token].append(docId)
        else:
            temp_post_list = vocab[token]
            temp_post_list.append(docId)
            # prevent duplication
            temp_post_list = sorted(set(temp_post_list))
            # back to linkedlist
            temp_post_list = deque(temp_post_list)
            vocab[token] = temp_post_list
    vocab_freq = {}
    # pairing term and doc frequency
    for key, val in tqdm(vocab.items()):
        vocab_freq[(key, len(val))] = val
    return vocab_freq


class BooleanQueries:
    
    def __init__(self, inverted_index: Dict[Tuple[str, int], Deque[int]], debug=False):
        self.inverted_index = inverted_index
        self.debug = debug
        keys_of_inverted_index = self.inverted_index.keys()
        self.keys_of_inverted_index = dict(keys_of_inverted_index)
    
    def search(self, query: str) -> Deque[int]:
        doc_freq: int = self.keys_of_inverted_index[query]
        pl_q: Deque[int] = self.inverted_index[(query, doc_freq)]
        return pl_q

    def intersection_query(self, query1, query2):
        # get doc frequency from query
        doc_freq1 = self.keys_of_inverted_index[query1]
        doc_freq2 = self.keys_of_inverted_index[query2]
        # get posting list
        pl_q1 = self.inverted_index[(query1, doc_freq1)]
        pl_q2 = self.inverted_index[(query2, doc_freq2)]
        answer = set(pl_q1) & set(pl_q2)
        if self.debug:
            data = [[pl_q1, pl_q2, answer]]
            headers = [f"posting list of {query1}", f"posting list of {query2}", "intersection result"]
            print(tabulate(data, headers=headers))
        return answer
    
    def union_query(self, query1, query2):
        # get doc frequency from query
        doc_freq1 = self.keys_of_inverted_index[query1]
        doc_freq2 = self.keys_of_inverted_index[query2]
        # get posting list
        pl_q1 = self.inverted_index[(query1, doc_freq1)]
        pl_q2 = self.inverted_index[(query2, doc_freq2)]
        answer = set(pl_q1) | set(pl_q2)
        if self.debug:
            data = [[pl_q1, pl_q2, answer]]
            headers = [f"posting list of {query1}", f"posting list of {query2}", "union result"]
            print(tabulate(data, headers=headers))
        return answer
    
    def negation_query(self, query):
        # get doc frequency from query
        doc_freq = self.keys_of_inverted_index[query]
        # get posting list
        pl_q = self.inverted_index[(query, doc_freq)]
        # get docID membership
        self.docID_membership()
        answer = set(self.docID_space) - set(pl_q)
        if self.debug:
            data = [[pl_q, self.docID_space , answer]]
            headers = [f"posting list of {query}", "docID membership", "negation result"]
            print(tabulate(data, headers=headers))
        return answer
    
    def docID_membership(self):
        docID_space = set()
        for posting_lists in self.inverted_index.values():
            for pl in posting_lists:
                docID_space.add(pl)
        self.docID_space = sorted(docID_space)


if __name__ == "__main__":
    pass
