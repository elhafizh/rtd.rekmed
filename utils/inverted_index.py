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


# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
# nltk.download('punkt')
nlp_id = spacy.blank('id')


def tokenization(contents):
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
                clean_token.append(token_of_nlp_contents.text)
        list_tokens_from_docs.append(clean_token)
    return list_tokens_from_docs



if __name__ == "__main__":
    pass
