from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import LJPDataset
from torch.utils.data import random_split
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score, accuracy_score
from utils import article2idx, penalty2idx

def get_charge2idx():
    with open("data/charge2idx.json", "r", encoding="utf-8") as fp:
        data = json.loads(fp.readline())
    return data
charge2idx = get_charge2idx()
idx2charge = {v: k for k, v in charge2idx.items()}
article2idx = article2idx
idx2article = {v: k for k, v in article2idx.items()}
penalty2idx = penalty2idx
idx2penalty = {v: k for k, v in penalty2idx.items()}

# 只要改这两个参数即可
task_type = "articles" # charges/articles/penalties
# model_pth = f"outputs/multi-defants/bert_{task_type}.pkl"
model_pth = f"outputs/multi-defants/bert_{task_type}.pkl"
preds_pth = f"predictions/{task_type}/{task_type}.jsonl"
model = torch.load(model_pth).to("cuda")
tokenizer = AutoTokenizer.from_pretrained('model/bert-base-chinese')
device = 'cuda'
texts = []

test_ds = []
with open('data/first_stage_test_300.jsonl', 'r', encoding='utf-8') as fp:
    for line in fp.readlines():
        data = json.loads(line)
        test_ds.append(data)
        id = data['id']
        fact = data['fact']
        defendants = data['defendants']
        prompt = "被告人是:"
        input_text = fact + prompt + defendants[0]
        texts.append(input_text)

def convert_preds(preds):
    results = []
    print(f"preds.shape={preds.shape}")
    for pred in preds:
        # charge = [idx2charge[idx] for idx, label in enumerate(pred) if label == True]
        if task_type == 'charges':
            charge = []
            for idx, label in enumerate(pred):
                if label == True:
                    charge.append(idx2charge[idx])
            results.append(charge)

        if task_type == 'articles':
            article = []
            for idx, label in enumerate(pred):
                if label == True:
                    article.append(idx2article[idx])
            results.append(article)

        if task_type == 'penalties':
            results.append(pred.tolist())

    return results

if task_type == "charges":

    inputs = tokenizer(texts, padding='max_length', max_length=512, return_tensors='pt', truncation=True).to(device)
    charges = []
    with torch.no_grad():
        logits = model(**inputs)['logits']
        preds = F.softmax(logits, dim=-1) > 0.5
        charges = convert_preds(preds)

    with open('predictions/charges/charges.jsonl', 'w', encoding='utf-8') as fp:
        for i, charge in enumerate(charges):
            data_js = {"id": i, "judgments": []}
            id = i
            num_defendants_case = len(test_ds[i]['defendants'])
            for defendant in test_ds[i]['defendants']:
                outcome = {"name": defendant, "charges": charge}
                data_js['judgments'].append(outcome)
            fp.writelines(json.dumps(data_js, ensure_ascii=False) + "\n")
    print(f"已经将{task_type}的结果写入到predictions/charges/charges.jsonl")

if task_type == 'articles':
    texts = []
    charges = []
    with open('predictions/charges/charges.jsonl', 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            data = json.loads(line)
            charge = data['judgments'][0]['charges']
            charges.append(charge)
    
    for data in test_ds:
        id = data['id']
        fact = data['fact']
        defendants = data['defendants']
        prompt = ["被告人: ", "被告人所犯的所有罪名如下："]
        text = fact + prompt[0] + defendants[0] + '。' + prompt[1]
        for charges_case in charges:
            for i, charge in enumerate(charges_case):
                if i < len(charges_case):
                    text = text + charge + ','
                else:
                    text = text + charge + '。'
        texts.append(text)
    inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)
    articles = []

    with torch.no_grad():
        logits = model(**inputs)['logits']
        preds = F.softmax(logits) > 0.5
        articles = convert_preds(preds)
        with open(preds_pth, 'w', encoding='utf-8') as fp:
            for i, article in enumerate(articles):
                data_js = {"id": i, "judgments": []}
                id = i
                num_defendants_case = len(test_ds[i]['defendants'])
                for defendant in test_ds[i]['defendants']:
                    outcome = {"name": defendant, "articles": article}
                    data_js['judgments'].append(outcome)
                fp.writelines(json.dumps(data_js, ensure_ascii=False) + "\n")
    print(f"已经将{task_type}的结果写入到predictions/{task_type}/{task_type}.jsonl")

if task_type == 'penalties':
    inputs = tokenizer(texts, padding='max_length', max_length=512, return_tensors='pt', truncation=True).to(device)
    penalties = []
    with torch.no_grad():
        logits = model(**inputs)['logits']
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        penalties = convert_preds(preds)

    with open(preds_pth, 'w', encoding='utf-8') as fp:
        for i, penalty in enumerate(penalties):
            data_js = {"id": i, "judgments": []}
            id = i
            num_defendants_case = len(test_ds[i]['defendants'])
            for defendant in test_ds[i]['defendants']:
                outcome = {"name": defendant, "penalties": penalty}
                data_js['judgments'].append(outcome)
            fp.writelines(json.dumps(data_js, ensure_ascii=False) + "\n")
    print(f"已经将{task_type}的结果写入到predictions/{task_type}/{task_type}.jsonl")

"""dt = test_ds[0]
fact = dt['fact']
with torch.no_grad():
    inputs = tokenizer(fact, padding='max_length', truncation=True, return_tensors='pt').to(device)
    logits = model(**inputs)['logits']
    print(f"logits = {logits}")
    preds = F.softmax(logits, dim=-1)
    print(preds)
    preds = preds > 0.3
    task_type = "penalties"
    penalties = convert_preds(preds)
    print(penalties)"""