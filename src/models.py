import torch
import torch.nn as nn
import torch.nn.functional as F

class CSA(nn.Module):

    def __init__(self, args):
        super(CSA, self).__init__()

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim)
        self.embedding.weight = nn.Parameter(args.embed, requires_grad=True)

        n_class = 2
        filter_sizes = [3, 4, 5]
        n_filters = 100
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, args.embed_dim))
            for fs in filter_sizes
        ])

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_class)


    def forward(self, text, **kwargs):
        conved = [self.relu(conv(self.embedding(text).unsqueeze(1))).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # 一定要用pooling？
        cat = torch.cat(pooled, dim=1)
        pred = self.fc(cat)

        return pred


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim)
        self.embedding.weight = nn.Parameter(args.embed, requires_grad=True)

        hidden_dim = 256
        output_dim = 2
        self.rnn = nn.RNN(args.embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text.transpose(0, 1))
        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))





class LossFunction(nn.Module):

    def __init__(self, args):
        super(LossFunction, self).__init__()

        self.loss_function = nn.CrossEntropyLoss()
        self.target = torch.tensor([0, 0], device=args.device)

    def forward(self, pred, label):
        target = self.target & 0
        target[label] = 1
        loss = self.loss_function(pred, label)
        correct = (torch.argmax(pred, dim=1) == label).float().sum()

        return loss, correct