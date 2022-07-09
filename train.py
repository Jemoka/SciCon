from transformers import GPT2Model, GPT2Tokenizer

import torch

import pandas as pd

import os

data = pd.read_csv("./okbuddyphd.csv", index_col=0)

model = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer("Pretrained model on English language using a causal language modeling (CLM) objective. It was introduced in this paper and first released at this page.")

