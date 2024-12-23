import json
from transformers import AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese')
max_length = 512

def get_charge2idx():
    with open("data/charge2idx.json", "r", encoding="utf-8") as fp:
        data = json.loads(fp.readline())
    return data
charge2idx = get_charge2idx()

def get_article2idx():
    with open("data/article2idx_paragraphs.json", 'r', encoding='utf-8') as fp:
        data = json.loads(fp.readline())
    return data
article2idx = get_article2idx()

def get_penalty2idx():
    penalty2idx = {
        "死刑": 0,
        "无期徒刑": 1,
        "拘役":2,
        "管制一年以上": 3,
        "管制一年以下": 4,
        "有期徒刑10年以上": 5,
        "有期徒刑7年以上，10年以下": 6,
        "有期徒刑5年以上，7年以下": 7,
        "有期徒刑3年以上，5年以下": 8,
        "有期徒刑2年以上，3年以下": 9,
        "有期徒刑1年以上，2年以下": 10,
        "有期徒刑9个月以上，1年以下": 11,
        "有期徒刑6个月以上，9个月以下": 12,
        "有期徒刑6个月以下": 13,
        "免予刑事处罚": 14
    }
    return penalty2idx
penalty2idx = get_penalty2idx()