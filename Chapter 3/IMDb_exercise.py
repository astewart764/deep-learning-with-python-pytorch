import torch
import numpy as np
import matplotlib as plt
import torchtext as torchtx
import pandas as pd
from pathlib import Path
import os
import re

def VectoriseStringData(str_dir, vec_dir, dict_in):
    count = 0
    for file in os.listdir(str_dir):
        string = open(str_dir + "\\" + file, encoding = 'utf-8', mode = 'r')
        split_string = re.split('[\s*`,":;.<>\/!?()]', string.read())

        for idx, value in enumerate(split_string):
            if value.endswith("'"):
                rem = -1
            elif value.endswith("'s"):
                rem = -2
            else:
                rem = len(value)
            split_string[idx] = value[:rem]

        # split_string = [value[:-1] for value in split_string if value.endswith("'")]
        joined_string = " ".join([i.lower() for i in split_string if i not in ["", "br"]])
        print(joined_string)

        '''
            find a way to convert numbers to text-based numbering (e.g. 35 to thirty-five) maybe check Stackoverflow for solution
        '''

        vectorised_string = [dict_in[stringval] for stringval in joined_string.split(" ")]
        print(vectorised_string)
        string.close()

        if count < 20:
            count += 1
        else:
            break

# IMDB_dataset = torchtx.datasets.IMDB(root = '../.data/')

ratings = pd.read_csv('../.data/IMDB/aclImdb/imdbEr.txt', sep = " ", header = None, dtype = np.float32)
ratings.columns = ["word_rating"]

vocab = pd.read_csv('../.data/IMDB/aclImdb/imdb.vocab', sep = " ", header = None, dtype = np.str)
vocab.columns = ["word"]

voc_scores = pd.concat([vocab, ratings], axis = 1, join = 'inner')
print(voc_scores.head())

voc_scores = voc_scores[:30000]
voc_scores.to_csv('../.data/IMDB/Custom/vocab_scores.csv')

vocab_scores = pd.read_csv('../.data/IMDB/Custom/vocab_scores.csv')
vocab_ratings = dict([(row["word"], row["word_rating"]) for label, row in vocab_scores.iterrows()])
vocab_vectors = dict([(row["word"], index) for index, row in vocab_scores.iterrows()])

VectoriseStringData('../.data/IMDB/aclImdb/train/pos', '../.data/IMDB/Custom/train/pos', vocab_vectors)
