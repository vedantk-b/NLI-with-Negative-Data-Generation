import numpy as np
import pandas as pd
import torch
from datasets import Dataset

train_df = pd.read_excel("train.xlsx")

eval_df = pd.read_excel("evaluation.xlsx")

def get_data():
    return train_df, eval_df

def get_train_eval_data():
    neglis = []
    with open('neglis.txt', 'r') as f:
        for line in f.readlines():
            neglis.append(line.replace(",", "").rstrip())
    negres = []
    with open('negres.txt', 'r') as f:
        for line in f.readlines():
            negres.append(line.replace(",", "").rstrip())

    neg_df = pd.DataFrame({"text":train_df["text"], "reason":neglis, "label": 0})
    negres_df = pd.DataFrame({"text":train_df["text"], "reason":negres, "label": 0})

    pos_df = train_df
    train_df = pd.concat((pos_df, neg_df, negres_df), axis = 0)

    train_data = Dataset.from_pandas(train_df)
    train_data = train_data.remove_columns("__index_level_0__")
    eval_data = Dataset.from_pandas(eval_df)

    return train_data, eval_data

