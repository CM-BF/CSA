import csv
import argparse
import torch
import torchtext.data as data
from torchtext import vocab
import execute
import models
from utils import tokenizer, set_min_len, tokenizer_len, tokenizer_bert, model_name
import random
import os
from transformers import BertTokenizer


random.seed(123)

parser = argparse.ArgumentParser(description='Sentimental Analysis Classification.')
parser.add_argument('--model', type=str, default='CSA', help='Choose the model.')
parser.add_argument('--test', action='store_true', help='Flag: for testing model.')
parser.add_argument('--test_ip', action='store_true', help='Flag: for testing while interpret the model.')
parser.add_argument('--interpret', action='store_true', help='Flag: for model interpreting.')
parser.add_argument('--epoch', type=int, default=100, help='Epochs to run.\n Default: 10')
parser.add_argument('--init_lr', type=float, default=1e-2, help='Init learning rate.\n Default: 1e-3')
parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help='Path: Save model checkpoint.')
parser.add_argument('--checkpoint', type=str, default=None, help='Path: Load model checkpoint.')
parser.add_argument('--dataset_dir', type=str, default='/home/citrine/datasets/SA/', help='Path of datasets.')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset.')
args = parser.parse_args()
args.start = 0
args.min_len = 5
set_min_len(args.min_len)
if torch.cuda.is_available():
    args.cuda = True
    args.device = torch.device('cuda')
else:
    args.cuda = False
    args.device = torch.device('cpu')

args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model, args.dataset_name)
print('checkpoint_dir:', args.checkpoint_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.system('mkdir -p %s' % args.checkpoint_dir)

# args.seq_length = 300
# bert: add use_vocab=False
if args.model == 'BERT':
    args.bert_tokenizer = tokenizer_bert.from_pretrained(model_name)
    TEXT = data.Field(sequential=True, tokenize=args.bert_tokenizer.encode_custom, include_lengths=True, use_vocab=False,
                      pad_token=args.bert_tokenizer.pad_token_id,
                      unk_token=args.bert_tokenizer.unk_token_id)
else:
    TEXT = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True)  # , fix_length=args.seq_length)
LABEL = data.Field(sequential=False, use_vocab=False)

# train, val, test = data.TabularDataset.splits(path='/home/citrine/datasets/SA/',
#                                               train='train.csv',
#                                               validation='val.csv',
#                                               test='test.csv',
#                                               format='csv', fields=[('Label', LABEL), ('Text', TEXT)])
custom_dataset = data.TabularDataset.splits(path=args.dataset_dir, train=args.dataset_name,
                                            format='csv', fields=[('Label', LABEL), ('Text', TEXT)])[0]
train_val, test = custom_dataset.split(0.8)
# train_val, test = custom_dataset.split(0.1)
train, val = train_val.split(7.0 / 8.0)


if 'BERT' not in args.model:
    wordvec = vocab.Vectors(name='Tencent_AILab_ChineseEmbedding.txt', cache='/home/citrine/datasets/wordvec/')
    TEXT.build_vocab(train, val, test, vectors=wordvec)
    args.TEXT = TEXT
    args.embed = TEXT.vocab.vectors
    args.embed_num = len(TEXT.vocab)
    args.embed_dim = TEXT.vocab.vectors.shape[1]


train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(16, 128, 1),
    sort_key=lambda x: len(x.Text),
    sort_within_batch=True, device=args.device)


if args.model == 'CSA':
    model = models.CSA(args)
elif args.model == 'LSTM':
    model = models.LSTM(args)
elif 'BERT' in args.model:
    model = models.BERT()
loss_function = models.LossFunction(args)
if args.cuda:
    model = model.cuda()
    loss_function = loss_function.cuda()
if args.test or args.interpret:
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['state_dict'])
    else:
        raise Exception('Not checkpoint available.')

if args.interpret:
    model.eval()  # for calling backward
    while 1:
        print('please input:', end='')
        text = input()
        if text == 'q':
            break
        if 'BERT' in args.model:
            text = torch.tensor([args.bert_tokenizer.encode_custom(text)], device=args.device)
            text_lengths = len(text)
        else:
            text, text_lengths = tokenizer_len(text)
            text = [args.TEXT.vocab.stoi[t] for t in text]
            text = torch.tensor(text, device=args.device).unsqueeze(0)
        text_lengths = torch.tensor([text_lengths], device=args.device)
        execute.interpret_sentence(model, text, text_lengths, args)

if args.test:
    execute.test(test_iter, model, loss_function, args)
else:
    execute.train(train_iter, val_iter, model, loss_function, args)
