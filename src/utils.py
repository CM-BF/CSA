import jieba
import re
from transformers import BertTokenizer

model_name = 'hfl/chinese-bert-wwm-ext'

min_len = 5

def set_min_len(x):
    global min_len
    min_len = x


def tokenizer(text):
    global min_len
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9，。！,.!\]\[@]')
    text = regex.sub(' ', text)
    tokenized = [word for word in jieba.cut(text) if word.strip()]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))

    return tokenized

def tokenizer_token(text):
    global min_len
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9，。！,.!\]\[@]')
    text = regex.sub(' ', text)
    tokenized = [word for word in text if word.strip()]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))

    return tokenized

def tokenizer_token_len(text):
    global min_len
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9，。！,.!\]\[@]')
    text = regex.sub(' ', text)
    tokenized = [word for word in text if word.strip()]
    text_lengths = len(tokenized)
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))

    return tokenized, text_lengths


def tokenizer_len(text):
    global min_len
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9，。！,.!]')
    text = regex.sub(' ', text)
    tokenized = [word for word in jieba.cut(text) if word.strip()]
    text_lengths = len(tokenized)
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))

    return tokenized, text_lengths


class tokenizer_bert(BertTokenizer):

    def encode_custom(self, text):
        ret = self.encode(text, max_length=100)
        if ret[-1] == self.vocab['[SEP]']:
            ret = ret[:-1]
        return ret