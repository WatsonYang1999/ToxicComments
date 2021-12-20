import torch.utils.data
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader
import numpy as np
from utils import *


def read_comments(data_dir, is_train):
    import pandas as pd

    df = pd.read_csv(data_dir)
    print(df.columns)
    data = df['comment_text']
    if is_train:
        target = df['target']
    else:
        target = None
    data = data.tolist()
    index = df['id'].tolist()
    if target is not None: target = target.tolist()
    for comment in data:
        comment = comment.replace('\n', ' ')
    return data, target, index


class CommentDataset(Dataset):
    def __init__(self, comments, targets, index):
        super(Dataset, self).__init__()
        self.comments = comments
        self.targets = targets
        self.index = index

    def __getitem__(self, item):
        if self.targets is None:
            return self.comments[item], self.index[item]

        return self.comments[item], self.targets[item], self.index[item]

    def __len__(self):
        return len(self.comments)


def get_weight(x: float):
    return int(x * 10)


data_dir = "NLP_Dataset/MLHomework_Toxicity/train.csv"
max_seq_len = 500
data_train, target_train, index_train = read_comments(data_dir, True)
train_tokens = tokenize(data_train, token='word')
vocab = Vocab(train_tokens, min_freq=2, reserved_tokens=['<pad>'])

train_features = torch.tensor([
    truncate_pad(vocab[line], max_seq_len, vocab['<pad>'])
    for line in train_tokens])

dataset_train = CommentDataset(train_features, target_train, index_train)
weights = []
for idx, (c, t, i) in enumerate(dataset_train):
    weights.append(t)
train_sampler = WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(
    dataset=dataset_train,
    sampler=train_sampler,
    shuffle=False,
    batch_size=100
)
toxicity_list = []
for idx, batch in enumerate(train_loader):
    _, t, i = batch
    t = t.detach().numpy().tolist()
    toxicity_list = toxicity_list+t
    print(i)

import matplotlib.pyplot as plt

plt.hist(
    x = toxicity_list,
    bins = 20
)
plt.show()
