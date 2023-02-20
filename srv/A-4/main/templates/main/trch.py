import torch
import transformers, emoji, soynlp, pytorch_lightning
import os
import pandas as pd
import numpy as np

from pprint import pprint

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from pytorch_lightning import LightningModule, Trainer, seed_everything

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize

import kss

args = {
    'random_seed': 42, # Random Seed
    'pretrained_model': 'beomi/KcELECTRA-base',  # Transformers PLM name
    'pretrained_tokenizer': '',  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    'batch_size': 32,
    'lr': 5e-6,  # Starting Learning Rate
    'epochs': 300,  # Max Epochs
    'max_length': 64,  # Max Length input size
    'train_data_path': '/content/drive/MyDrive/train_6000_2.csv',  # Train Dataset file 
    'val_data_path': '/content/drive/MyDrive/valid_6000_2.csv',  # Validation Dataset file 
    'test_mode': False,  # Test Mode enables `fast_dev_run`
    'optimizer': 'AdamW',  # AdamW vs AdamP
    'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
    'fp16': True,  # Enable train on FP16(if GPU)
    'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
    # 'cpu_workers': os.cpu_count(),
    'cpu_workers': 0
}

accuracy_list = []
loss_list = []
class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() # 이 부분에서 self.hparams에 위 kwargs가 저장된다.
        
        self.clsfier = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels = 5)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, **kwargs):
        return self.clsfier(**kwargs)

    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average = 'micro')
        rec = recall_score(y_true, y_pred, average = 'micro')
        f1 = f1_score(y_true, y_pred, average = 'micro')

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', acc, on_epoch=True, prog_bar=True)
        self.log(state+'_precision', prec, on_epoch=True, prog_bar=True)
        self.log(state+'_recall', rec, on_epoch=True, prog_bar=True)
        self.log(state+'_f1', f1, on_epoch=True, prog_bar=True)
    
        accuracy_list.append(acc)
        loss_list.append(loss)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}, Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay = 0.3)
            # optimizer = AdamW(self.parameters(), lr=0)#0에 가까운 아주 작은 값을 입력해야함
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        elif self.hparams.lr_scheduler == 'cosup':
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=0.1,  T_up=10, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def clean(self, x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    def encode(self, x, **kwargs):
        return self.tokenizer.encode(
            self.clean(str(x)),
            padding='max_length',
            max_length=self.hparams.max_length,
            truncation=True,
            **kwargs,
        )

    def preprocess_dataframe(self, df):
        df['document'] = df['document'].map(self.encode)
        return df

    def dataloader(self, path, shuffle=False):
        df = self.read_data(path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['document'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size * 1 if not self.hparams.tpu_cores else self.hparams.tpu_cores,
            shuffle=shuffle,
            num_workers=self.hparams.cpu_workers,
        )

    def train_dataloader(self):
        return self.dataloader(self.hparams.train_data_path, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.hparams.val_data_path, shuffle=False)
        
def infer(x):
    return torch.softmax(
        model(**model.tokenizer(x, return_tensors='pt')
    ).logits, dim=-1)

device = torch.device('cuda')
print("load model")
model = Model(**args)
model.load_state_dict(torch.load('/srv/A-4/final_model.pt'))
model.eval()

# text = "정말 옛날옛날에 애용하던 러쉬에 오랜만에 가봤다. 다시 보니 또 추억 뿜뿜 몇 가지 들고 집으로 왔다:) 거의 매달 색을 바꾸니 헤어가 남아나질 않...ㅠ 하이라이트를 다음엔 꼭 하리라!ㅋ 혼자만의 시간을 보낸 론 헤르만 카페 정말 너무 예쁘고, 맛있는 곳완전 추천추천!! 유치원 갔다오는 회장님과 함께 또 돈카츠 먹기ㅎㅎ 맛있는 점심을 꼬기꼬기 먹고, 꼬기 사진은 유툽용으로 동영상만 찍었더니 하나도 없다ㅠ 주변 공원으로 가서 회장님 자전거 연습:) 오늘이 추석이라는 사실도 잊고 있었다.. 운동 간단히 하고 나오는데, 보름달이 너무 예뻤다:)"

# sp_text = kss.split_sentences(text)

# import numpy as np
def emotion_count(li):
  counts = {}
  for x in li:
    if x in counts:
      counts[x] += 1
    else:
      counts[x] = 1
  return counts

def top_3(count_dict, n = 3):
  return sorted(count_dict.items(), reverse = True, key = lambda x:x[1])[:n]

def neu_emo(text):
  sec_emotions = []
  for sent in text:
    single_text = infer(sent)

    lo = []

    for i in single_text:
      a = i.detach().numpy()
      a = a.tolist()
      lo.append(a)
      lo = np.concatenate(lo).tolist()
      lo_sor = sorted(lo)
      sec_max_emo = lo.index(lo_sor[-2])

      if sec_max_emo == 0:
        sec_emotions.append("불안")
      elif sec_max_emo == 1:
        sec_emotions.append("중립")
      elif sec_max_emo == 2:
        sec_emotions.append("분노")
      elif sec_max_emo == 3:
        sec_emotions.append("슬픔")
      elif sec_max_emo == 4:
        sec_emotions.append("행복")    

  counts = emotion_count(sec_emotions)
  top = top_3(counts, n = 3)
  return top
#시작

def infer_re(text):
  sp_text = kss.split_sentences(text)
  max_emotions = []
  for sent in sp_text:
    single_text = infer(sent)
    lo = []

    for i in single_text:
      a = i.detach().numpy()
      a = a.tolist()
      lo.append(a)
      lo = np.concatenate(lo).tolist()
      max_emo = lo.index(max(lo))

      if max_emo == 0:
        max_emotions.append("불안")
      elif max_emo == 1:
        max_emotions.append("중립")
      elif max_emo == 2:
        max_emotions.append("분노")
      elif max_emo == 3:
        max_emotions.append("슬픔")
      elif max_emo == 4:
        max_emotions.append("행복")    

  counts = emotion_count(max_emotions)
  top = [item[0] for item in top_3(counts, n = 3)]
  first_emo = top[0]
  print(top, first_emo)

  if len(top) == 1 and first_emo == '중립':
    top = [item[0] for item in neu_emo(sp_text)]
    first_emo = top[0]
    if len(top) == 1:#neu_emo돌려도 값 하나일 때
      return [first_emo]

  elif '중립' in top:
    top.remove('중립')
    return top
  
  else:
    return top