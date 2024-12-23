from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
from utils import get_charge2idx
import torch
import numpy as np
from utils import article2idx

"""
    facts
    defendants
    charges

"""
class LJPDataset(Dataset):

    def __init__(self, tkz_pth, max_length=512, data_pth=None) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tkz_pth)
        self.max_length = max_length
        self.data_pth = data_pth
        self.charge2idx = get_charge2idx()
        self.article2idx = article2idx
        self.facts, self.defendants, self.charges, self.articles, self.penalties = self.load_data()
    
    def __getitem__(self, index):
        return self.facts[index], self.defendants[index], self.charges[index], self.articles[index], self.penalties[index]
    
    def __len__(self):
        return len(self.facts)
    
    def collate_fn_charges(self, batch):
        facts = [item[0] for item in batch]
        defendants_facts = [item[1] for item in batch]
        charges_defendants_facts = [item[2] for item in batch]
        inputs = []
        prompt = "被告人是:"
        for fact, defendants_fact in zip(facts, defendants_facts):
            for defendant_fact in defendants_fact:
                input = fact + prompt + defendant_fact
                inputs.append(input)
        targets = []
        for charges_defendants_fact in charges_defendants_facts:
            for charges_defendant_fact in charges_defendants_fact:
                target = np.zeros(len(self.charge2idx))
                label_target = []
                for charge_defendant_fact in charges_defendant_fact:
                    label_target.append(self.charge2idx[charge_defendant_fact])
                target[label_target] = 1
                targets.append(target)
        inputs = self.tokenizer(inputs, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        # np.array is for quickly converting
        targets = torch.tensor(np.array(targets), dtype=torch.float32)
        return inputs, targets
    
    def collate_fn_articles(self, batch):
        facts = [item[0] for item in batch]
        defendants_facts = [item[1] for item in batch]
        charges_defendants_facts = [item[2] for item in batch]
        articles_defendants_facts = [item[3] for item in batch]
        prompt = ["被告人: ", "被告人所犯的所有罪名如下："]
        inputs = []
        for fact, defendants_fact, charges_defendants_fact, in zip(facts, defendants_facts, charges_defendants_facts):
            for defendant_fact, charges_defendant_fact in zip(defendants_fact, charges_defendants_fact) :
                input = fact + prompt[0] + defendant_fact + ',' + prompt[1]
                for i, charge_defendant_fact in enumerate(charges_defendant_fact):
                    input = input + charge_defendant_fact
                    if i != len(charges_defendant_fact):
                        input = input + ','
                    else:
                        input = input + '。'
                inputs.append(input)
        
        targets = []
        for articles_defendants_fact in articles_defendants_facts:
            for articles_defendant_fact in articles_defendants_fact:
                target = [0] * len(self.article2idx)
                for article_defendant_fact in articles_defendant_fact:
                    target[self.article2idx[article_defendant_fact]] = 1
                targets.append(target)
        # print(f"len(facts)={len(facts)}, len(inputs)={len(inputs)}, len(targets)={len(targets)}")
        inputs = self.tokenizer(inputs, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        targets = torch.tensor(targets, dtype=torch.float32)
        return inputs, targets

    def collate_fn_penalty(self, batch):
        facts = [item[0] for item in batch]
        defendants_facts = [item[1] for item in batch]
        penalties_defendants_facts = [item[4] for item in batch]
        prompt = "被告人是:"
        
        inputs = []
        for fact, defendants_fact in zip(facts, defendants_facts):
            for defendant_fact in defendants_fact:
                input = fact + prompt + defendant_fact
                inputs.append(input)
        
        targets = []
        for penalties_defendants_fact in penalties_defendants_facts:
            for penalty_defendant_fact in penalties_defendants_fact:
                target = penalty_defendant_fact
                targets.append(target)
                
        inputs = self.tokenizer(inputs, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        targets = torch.tensor(targets, dtype=torch.int64)
        return inputs, targets

    def load_data(self):
        # 对每一个case的列表
        facts = [] # 案件事实列表
        defendants_facts = [] # A list of defendants of a case 被告人列表
        charges_defendants_facts = [] # A list of charges of all defendants of a case 被告人罪名列表
        articles_defendants_facts = [] # 被告人触犯法条列表
        penalties_defendants_facts = [] # 被告人判决结果列表

        with open(self.data_pth, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                case = json.loads(line)
                fact = case["fact"]
                outcomes = case["outcomes"]
                # Everyone has an relative outcome
                defendants_fact = []
                charges_defendants_fact = []
                articles_defendants_fact = []
                penalties_defendants_fact = []
                for outcome in outcomes:
                    defendant = outcome['name']
                    defendants_fact.append(defendant)
                    charges_defendant_fact = outcome['charges']
                    charges_defendants_fact.append(charges_defendant_fact)
                    articles_defendant_fact = outcome['articles']
                    articles_defendants_fact.append(articles_defendant_fact)
                    penalty = outcome['penalty']
                    penalty_defendant_fact = None
                    if penalty['surveillance'] != 0:
                        penalty_defendant_fact = 2
                    if penalty['detention'] > 0 and penalty['detention'] <= 12:
                        penalty_defendant_fact = 4
                    else:
                        penalty_defendant_fact = 3
                    if penalty['imprisonment'] > 0 and penalty['imprisonment'] < 6:
                        penalty_defendant_fact = 13
                    if penalty['imprisonment'] >= 6 and penalty['imprisonment'] < 9:
                        penalty_defendant_fact = 12
                    if penalty['imprisonment'] >= 9 and penalty['imprisonment'] < 12:
                        penalty_defendant_fact = 11
                    if penalty['imprisonment'] >= 12 and penalty['imprisonment'] < 24:
                        penalty_defendant_fact = 10
                    if penalty['imprisonment'] >= 24 and penalty['imprisonment'] < 36:
                        penalty_defendant_fact = 9
                    if penalty['imprisonment'] >= 36 and penalty['imprisonment'] < 60:
                        penalty_defendant_fact = 8
                    if penalty['imprisonment'] >= 60 and penalty['imprisonment'] < 84:
                        penalty_defendant_fact = 7
                    if penalty['imprisonment'] >= 84 and penalty['imprisonment'] < 120:
                        penalty_defendant_fact = 6
                    if penalty['imprisonment'] >= 120:
                        penalty_defendant_fact = 5
                    if penalty['death_penalty'] == True:
                        penalty_defendant_fact = 0
                    if penalty['life_imprisonment'] == True:
                        penalty_defendant_fact = 1
                    if penalty['surveillance'] == 0 and  penalty['life_imprisonment'] == False and penalty['detention'] == 0 and penalty['imprisonment'] == 0 and penalty['death_penalty'] == False:
                        penalty_defendant_fact = 14

                    penalties_defendants_fact.append(penalty_defendant_fact)
                
                facts.append(fact)
                defendants_facts.append(defendants_fact)
                charges_defendants_facts.append(charges_defendants_fact)
                articles_defendants_facts.append(articles_defendants_fact)
                penalties_defendants_facts.append(penalties_defendants_fact)

        return facts, defendants_facts, charges_defendants_facts, articles_defendants_facts, penalties_defendants_facts