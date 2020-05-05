import torch
from transformers import *

model_name = 'hfl/chinese-bert-wwm-ext'

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

vocab = tokenizer.vocab
print()