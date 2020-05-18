import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from utils import model_name


class CNN(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()

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

    def forward(self, text, text_length, **kwargs):
        conved = [self.relu(conv(self.embedding(text).unsqueeze(1))).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # 一定要用pooling？
        cat = torch.cat(pooled, dim=1)
        pred = self.fc(cat)

        return pred


class CNN_5(nn.Module):

    def __init__(self, args):
        super(CNN_5, self).__init__()

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim)
        self.embedding.weight = nn.Parameter(args.embed, requires_grad=True)

        n_class = 2
        filter_sizes = [3]
        n_filters = 100
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, args.embed_dim))
            for fs in filter_sizes
        ])

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_class)

    def forward(self, text, text_length, **kwargs):
        conved = [self.relu(conv(self.embedding(text).unsqueeze(1))).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # 一定要用pooling？
        cat = torch.cat(pooled, dim=1)
        pred = self.fc(cat)

        return pred


class TCNN(nn.Module):

    def __init__(self, args):
        super(TCNN, self).__init__()

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim)
        self.embedding.weight = nn.Parameter(args.embed, requires_grad=True)

        n_class = 2
        filter_sizes = [3, 4, 5]
        n_filters = args.embed_dim

        self.transformers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=args.embed_dim, nhead=10),
            nn.Tanh()
        )
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, args.embed_dim))
            for fs in filter_sizes
        ])

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_class)

    def forward(self, text, text_length, **kwargs):
        embed_text = self.embedding(text) # [N, S, D]
        weighted_map = self.transformers(embed_text) # multiply residual net
        weighted_text = embed_text * weighted_map
        # print(weighted_map)

        conved = [self.relu(conv(weighted_text.unsqueeze(1))).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # 一定要用pooling？
        cat = torch.cat(pooled, dim=1)
        pred = self.fc(cat)

        return pred

class TCNN_n(nn.Module):

    def __init__(self, args):
        super(TCNN_n, self).__init__()

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim)
        self.embedding.weight = nn.Parameter(args.embed, requires_grad=True)

        n_class = 2
        filter_sizes = [3, 4, 5]
        n_filters = args.embed_dim

        self.transformers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=args.embed_dim, nhead=10),
            nn.Tanh()
        )
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, args.embed_dim))
            for fs in filter_sizes
        ])

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_class)

    def forward(self, text, text_length, **kwargs):
        embed_text = self.embedding(text) # [N, S, D]
        weighted_map = self.transformers(embed_text) # multiply residual net
        weighted_text = weighted_map

        conved = [self.relu(conv(weighted_text.unsqueeze(1))).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # 一定要用pooling？
        cat = torch.cat(pooled, dim=1)
        pred = self.fc(cat)

        return pred


class TCNN_p(nn.Module):

    def __init__(self, args):
        super(TCNN_p, self).__init__()

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim)
        self.embedding.weight = nn.Parameter(args.embed, requires_grad=True)

        n_class = 2
        filter_sizes = [3, 5, 7]
        n_filters = args.embed_dim


        self.transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=args.embed_dim, nhead=10)
        ] * 3)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      padding=((fs - 1)//2, 0),
                      kernel_size=(fs, args.embed_dim))
            for fs in filter_sizes
        ])

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_class)

    def forward(self, text, text_length, **kwargs):
        embed_text = self.embedding(text) # [N, S, D]
        weighted_map = [self.relu(transformer(embed_text)).permute(0, 2, 1) for transformer in self.transformers] # multiply residual net

        conved = [self.relu(conv(embed_text.unsqueeze(1))).squeeze(3) for conv in self.convs]
        weighted_conved = [weighted_map[i] * conved[i] for i in range(len(conved))]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in weighted_conved]  # 一定要用pooling？
        cat = torch.cat(pooled, dim=1)
        pred = self.fc(cat)

        return pred


class TCNN_l(nn.Module):

    def __init__(self, args):
        super(TCNN_l, self).__init__()

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim)
        self.embedding.weight = nn.Parameter(args.embed, requires_grad=True)

        n_class = 2
        filter_sizes = [3, 5, 7]
        n_filters = args.embed_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.embed_dim, nhead=10)
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, 1)
        ] * 3)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      padding=((fs - 1)//2, 0),
                      kernel_size=(fs, args.embed_dim))
            for fs in filter_sizes
        ])

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_class)

    def forward(self, text, text_length, **kwargs):
        embed_text = self.embedding(text) # [N, S, D]

        conved = [self.relu(conv(embed_text.unsqueeze(1))).squeeze(3).permute(0, 2, 1) for conv in self.convs]
        weighted_map = [self.relu(self.transformers[i](conved[i])) for i in range(len(conved))]  # multiply residual net

        weighted_conved = [(weighted_map[i] * conved[i]).permute(0, 2, 1) for i in range(len(conved))]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in weighted_conved]  # 一定要用pooling？
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
        n_layers = 2
        dropout = 0.5
        self.rnn = nn.LSTM(args.embed_dim, hidden_dim,
                           n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text).transpose(0, 1)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_input)
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden)


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        hidden_dim = 768
        output_dim = 2
        self.model = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        hidden, _ = self.model(text)
        pred = self.fc(hidden[:, 0])
        return pred


class LossFunction(nn.Module):

    def __init__(self, args):
        super(LossFunction, self).__init__()

        self.loss_function = nn.CrossEntropyLoss()
        self.target = torch.tensor([0, 0], device=args.device)

    def forward(self, pred, label):
        target = self.target & 0
        target[label] = 1
        loss = self.loss_function(pred, label)
        pred_label = torch.argmax(pred, dim=1)
        tp = (pred_label & label).float().sum()
        tn = ((pred_label ^ 1) & (label ^ 1)).float().sum()
        fp = (pred_label & (label ^ 1)).float().sum()
        fn = ((pred_label ^ 1) & label).float().sum()

        return loss, {'tp': tp.item(), 'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item()}
