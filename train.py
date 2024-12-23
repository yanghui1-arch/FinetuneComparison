import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer
from tqdm import tqdm
import time
from torch.optim import AdamW
from torch.nn import functional as F
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score, accuracy_score
from torch.utils.data import DataLoader
from dataset import LJPDataset
from torch.utils.data import random_split
import numpy as np
from utils import get_article2idx, get_charge2idx, get_penalty2idx

from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
import bitsandbytes as bnb
from torch.amp import GradScaler, autocast
from transformers import BitsAndBytesConfig
from model.p_bert.pbert import PBERT


"""
    modify task_type, model_pth and collate_fn when perform training.
"""
settings = {
    "tkz_pth": "model/bert-base-chinese",
    # "epochs": 10,
    "epochs": 20,
    "iter_steps": 1000,
    "records_path": "outputs/multi-defants",
    "ckp_path": "outputs/multi-defants",
    "train_data_pth": "data/first_stage_train_5000.jsonl",
    "test_data_pth": "data/first_stage_test_300.jsonl",
    "batch_size": 4,
    "device": "cuda",
    "model_pth": "model/bert-base-chinese",
    "task_type": "charges",
    "preds_threshold": 0.5,
    "finetune": "p-tuning" # LoRA, QLoRA, FF, p_tuning
}

torch.cuda.empty_cache()

generator = torch.Generator()
def set_seed():
    generator.manual_seed(3)

set_seed()
    

class Trainer:
    def __init__(self, model, train_dl, test_dl, optimizer: AdamW):
        self.tokenizer = AutoTokenizer.from_pretrained(settings["tkz_pth"])
        self.model = model.to(settings['device'])
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = "cuda"
        self.optimizer = optimizer
        self.cur_step = 1
        self.cur_min_loss = float("inf")

        if settings['finetune'] == "FF":
            self.scaler = GradScaler(enabled=True)
    
    def train(self, ):
        self.model.train()
        num_training_steps = settings['epochs'] * len(self.train_dl)
        self.process_bar = tqdm(range(num_training_steps))
        self.time_stamp = time.time()
        for epoch in range(settings['epochs']):
            self._run_epoch()
        self._save_ckp()

    def _run_epoch(self,):
        for inputs, charges in self.train_dl:
            self._run_batch(inputs, charges)
            if self.cur_step % settings['iter_steps'] == 0:
                self.model.eval()
                vl = self._run_eval()
                if vl < self.cur_min_loss:
                    self._save_ckp()
                    self.cur_min_loss = vl
                self.model.train()
                print(f"time consuming: {round((time.time()-self.time_stamp)/60,2)}min")
                self.time_stamp = time.time()
            self.cur_step += 1

    def _run_batch(self, inputs, charges):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        charges = charges.to(self.device)
        self.optimizer.zero_grad()

        if settings['finetune'] == 'FF':
            with autocast(device_type=settings['device'], enabled=True, dtype=torch.float16):
                output = self.model(**inputs)
                logits = output.logits
            
                if settings['task_type'] == 'penalties':
                    loss = F.cross_entropy(logits, charges)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, charges.float())

            self.scaler.scale(loss).backward()
            self.scaler.step(optimzer)
            self.scaler.update()

        else:
            logits = self.model(**inputs)['logits']
            
            if settings['task_type'] == 'penalties':
                loss = F.cross_entropy(logits, charges)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, charges.float())
            loss.backward()
            self.optimizer.step()

        self.process_bar.update(1)

    def _run_eval(self,):
        total_loss=0
        Y = []
        Y_hat = []
        with torch.no_grad():
            for inputs, charges in self.test_dl:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                charges = charges.to(self.device)
                logits = self.model(**inputs)["logits"]
                if settings['task_type'] == 'penalties':
                    loss = F.cross_entropy(logits, charges)
                    preds = torch.argmax(logits, dim=-1)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, charges.float())
                    preds = F.softmax(logits, dim=-1) > settings['preds_threshold']
                total_loss+=loss.item()
                Y.extend(charges.tolist())
                Y_hat.extend(preds.tolist())
        avg_method = "macro" if settings['task_type'] == 'penalties' else "samples"
        
        acc = f"Acc: {round(accuracy_score(y_pred=Y_hat, y_true=Y), 4)}"
        p = f"P: {round(precision_score(y_pred=Y_hat, y_true=Y, average=avg_method), 4)}"
        f1 = f"F1: {round(f1_score(y_pred=Y_hat, y_true=Y, average=avg_method), 4)}"
        r = f"R: {round(recall_score(y_pred=Y_hat, y_true=Y, average=avg_method), 4)}"
        avg_loss = round(total_loss/len(self.test_dl), 4)
        records = f"step: {self.cur_step+1} | loss: {avg_loss} | {acc} | {p} | {r} | {f1}"
        with open(settings["records_path"]+ "/" + f"{settings['task_type']}_train_records.txt", "a", encoding="utf-8") as fi:
            fi.write(records+"\n")
        print(records)
        return avg_loss

    def _save_ckp(self):
        # path = settings["ckp_path"] + "/" + f"{settings['task_type']}_step_{self.cur_step}_{settings['finetune']}.pkl"
        # torch.save(self.model, path)
        path = settings["ckp_path"] + "/" + f"{settings['task_type']}_{settings['finetune']}"
        self.model.save_pretrained(path)


if __name__ == "__main__":

    # charge2idx = get_charge2idx()
    # idx2charge = dict((v, k) for k, v in charge2idx.items())
    # article2idx = get_article2idx()
    # idx2article = dict((v, k) for k, v in article2idx.items())
    num_labels = None
    label2id = None
    id2label = None

    if settings['task_type'] == 'charges':
        label2id = get_charge2idx()
        id2label = dict((v, k) for k, v in label2id.items())
    if settings['task_type'] == 'articles':
        label2id = get_article2idx()
        id2label = dict((v, k) for k, v in label2id.items())
    # Need to fix
    if settings['task_type'] == 'penalties':
        label2id = get_penalty2idx()
        id2label = dict((v, k) for k, v in label2id.items())
    num_labels = len(label2id)

    # def __init__(self, tkz_pth, max_length, data_pth)
    train_ds = LJPDataset(tkz_pth=settings['tkz_pth'], data_pth=settings['train_data_pth'])

    if settings['task_type'] == "charges":
        collate_fn = train_ds.collate_fn_charges
    elif settings['task_type'] == 'articles':
        collate_fn = train_ds.collate_fn_articles
    else:
        collate_fn = train_ds.collate_fn_penalty

    train_ds, test_ds, val_ds = random_split(train_ds, [0.8, 0.1, 0.1], generator=generator)
    train_dl = DataLoader(train_ds, batch_size=settings['batch_size'], collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=settings['batch_size'], collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=settings['batch_size'], collate_fn=collate_fn)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        # settings['tkz_pth'], 
        "outputs/multi-defants/charges_QLoRA",
        num_labels=num_labels, 
        id2label=id2label, 
        label2id=label2id
    )
    
    if settings['finetune'] is not None:
        if settings['finetune'] == "LoRA":
            config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, config)

        if settings['finetune'] == "QLoRA":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                    settings['tkz_pth'], 
                    num_labels=num_labels, 
                    id2label=id2label, 
                    label2id=label2id,
                    quantization_config=quantization_config
            )
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, config)

        if settings['finetune'] == 'p-tuning':
            model = PBERT(model=model, prompt_tokens=10)
        
    # model = torch.load(settings['model_pth'])
    optimzer = AdamW(model.parameters(), lr=2e-5)
    trainer = Trainer(
        model=model, 
        optimizer=optimzer, 
        train_dl=train_dl, 
        test_dl=test_dl
    )
    
    trainer.train()