import jieba
import re

min_len = 0

def set_min_len(x):
    global min_len
    min_len = x


def tokenizer(text):
    global min_len
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9，。！,.!]')
    text = regex.sub(' ', text)
    tokenized = [word for word in jieba.cut(text) if word.strip()]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))

    return tokenized