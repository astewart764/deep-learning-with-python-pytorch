import torch
import numpy as np
import matplotlib as plt
import torchtext as torchtx
import pandas as pd
from pathlib import Path
import os
import re
from num2words import num2words

def CheckNum(single):
    flag = 0
    try:
        out = int(single)
    except:
        out = single
    else:
        out = num2words(single)
        flag = 1

    return(out, flag)

def VectoriseStringData(directory_in, file_in, dict_in):
        string = open(directory_in + "/" + file_in, encoding = 'utf-8', mode = 'r')
        split_string = re.split('[\s*`,":;.<>\/!?()~]', string.read())
        split_string = re.split('-', " ".join(split_string))

        for idx, value in enumerate(split_string):
            num_val, num_flag = CheckNum(value)

            if num_flag:
                split_string[idx] = num_val
            else:
                flag = 0
                if value.endswith("'"):
                    rem = -1
                    flag = 1
                elif value.endswith("'s"):
                    rem = -2
                    flag = 1
                else:
                    rem = len(value)

                if flag:
                    split_string[idx] = value[:rem]
                else:
                    continue

        # split_string = [value[:-1] for value in split_string if value.endswith("'")]
        joined_string = " ".join([i.lower() for i in split_string if i not in ["", "br"]])
        print(joined_string)
        vectorised_string = [dict_in[stringval] for stringval in joined_string.split(" ")]
        string.close()

# IMDB_dataset = torchtx.datasets.IMDB(root = '../.data/')

ratings = pd.read_csv('../.data/IMDB/aclImdb/imdbEr.txt', sep = " ", header = None, dtype = np.float32)
ratings.columns = ["word_rating"]

vocab = pd.read_csv('../.data/IMDB/aclImdb/imdb.vocab', sep = " ", header = None, dtype = np.str)
vocab.columns = ["word"]

voc_scores = pd.concat([vocab, ratings], axis = 1, join = 'inner')
print(voc_scores.head())

voc_scores = voc_scores
voc_scores.to_csv('../.data/IMDB/Custom/vocab_scores.csv')

vocab_scores = pd.read_csv('../.data/IMDB/Custom/vocab_scores.csv')
vocab_ratings = dict([(row["word"], row["word_rating"]) for label, row in vocab_scores.iterrows()])
vocab_vectors = dict([(row["word"], index) for index, row in vocab_scores.iterrows()])

file_directory = '../.data/IMDB/aclImdb/train/pos'
count = 0
for file in os.listdir(file_directory):
    print(file)
    VectoriseStringData(file_directory, file, vocab_vectors)

    if count == 20:
        break
    else:
        count += 1
