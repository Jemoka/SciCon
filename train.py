from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch

from torch.optim import AdamW

import pandas as pd

import os

from tqdm import tqdm

import random

import wandb

run = wandb.init(mode="disabled")

data = pd.read_csv("./okbuddyphd.csv", index_col=0)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

optim = AdamW(model.parameters(), lr=1e-5)

model.train()

for _ in range(3):

    for i in tqdm(data.iloc):

        res = model(**tokenizer(data.iloc[0]["posts"], return_tensors="pt", max_length=40, truncation=True).to(DEVICE), labels=tokenizer.encode(data.iloc[0]["posts"], return_tensors="pt").to(DEVICE))

        res["loss"].backward()
        optim.step()
        optim.zero_grad()

model.save_pretrained(run.name)

