# CSA

## Requirement

* pytorch 1.2
* torchtext
* other NLP packages (please refer to those error messages).

## Training
```python
python run.py --init_lr 0.001 --model [CSA/LSTM/BERT] --dataset_name your_dataset_name --dataset_dir your_dataset_directory 
```

Please modify line 67 in run.py to use your word vector file while not using BERT model.

## Testing

Base on the training command, adding flags `--test --checkpoint ../checkpoints/[CSA/LSTM/BERT]/CSA_?.ckpt` is enough.

