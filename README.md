# CSA
  
## Requirement

* pytorch 1.2
* torchtext
* [BERT model](https://github.com/ymcui/Chinese-BERT-wwm)
* other NLP packages (please refer to error messages).

## Word Vectors Loading

Click [Here](https://ai.tencent.com/ailab/nlp/zh/embedding.html) to download word vectors.
To simple the training script, the path to word vectors file is hard encoded in run.py line 84.

## Training

Before running, please locate your current work path at `src` directory.

```bash
$ python run.py
```

Flag Examples:

model: CNN/LSTM/TCNN_p, dataset: /A/B/weibo_senti_100k.csv

(I highly recommend you to hard encode `--dataset_dir` in run.py line 33. Datasets are included in datasets directory.)

```bash
--init_lr 0.001 --model CNN/LSTM/TCNN_p --dataset_name weibo_senti_100k.csv --dataset_dir /A/B
```

model: BERT, dataset: /A/B/weibo_senti_100k.csv

```bash
--init_lr 5e-5 --model BERT --dataset_name weibo_senti_100k.csv --dataset_dir /A/B
```


## Testing

```bash
python run.py
```

Flags:

```bash
--test --model CNN/LSTM/BERT/TCNN_p --dataset_name weibo_senti_100k.csv --dataset_dir /A/B
```

## interpret

Similarly, changing `--test` to `--interpret` can turn it into interpreting mode.

```bash
--interpret --model CNN/LSTM/BERT/TCNN_p --dataset_name weibo_senti_100k.csv --dataset_dir /A/B
```
