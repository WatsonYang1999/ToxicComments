import datetime

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model import TextCNN
from utils import *


class Timer():
    def __init__(self):
        self.begin_time = datetime.datetime.now()

    def query(self, stuff):
        current_time = datetime.datetime.now()
        print(("**************************************************"))
        print("\n")
        print(f"Time to {stuff} is {current_time - self.begin_time}")
        print("\n\n")
        print(("**************************************************"))


'''
加载评论，若为测试集则target为None
'''
def read_comments(data_dir, is_train):
    import pandas as pd

    df = pd.read_csv(data_dir)

    data = df['comment_text']
    print(df.shape)
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
        self.index = index # 数据对应的id，单纯最后为了方便输出submission

    def __getitem__(self, item):
        if self.targets is None:
            return self.comments[item], self.index[item]

        return self.comments[item], self.targets[item], self.index[item]

    def __len__(self):
        return len(self.comments)


def train(model, data_loaders, num_epochs, optimizer, device, loss):
    losses = []
    for i in range(0, num_epochs):
        for _i, batch in enumerate(data_loaders['train']):
            X, Y, _ = batch
            X = X.to(device)
            Y = Y.float().to(device)
            Y_hat = model(X)

            l = loss(Y_hat, Y)
            l.sum().backward()
            epoch_loss = l.sum().item()

            optimizer.step()
            losses.append(epoch_loss)
            optimizer.zero_grad()
            if _i % 100 == 0:
                print(f"training loss :   {epoch_loss}")
                validate(model, data_loaders['val'], loss, device)
            if device=='cpu' : print(epoch_loss)
        print("New Epoch Begin -------------------------------------")
        torch.save(model, 'model.pt')


def validate(model: torch.nn.Module, val_loader, loss, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, Y, index) in enumerate(val_loader):
            X = X.to(device)

            Y = Y.float().to(device)
            Y_hat = model(X)

            l = loss(Y_hat, Y)
            print(f"validating loss :   {l.sum().item()}")


def predict(model, test_loader):
    model.eval()
    '''
    照理来说应该用在验证集上表现最好的参数来提交，但本来训练也没多少轮，就算了。
    '''
    with open('submission.txt', 'w') as f:
        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                X, index = batch
                X = X.to(device)
                Y_hat = model(X)
                for i, idx in enumerate(index):
                    line = str(idx.item()) + " " + str(Y_hat[i].item()) + '\n'
                    f.write(line)


if __name__ == '__main__':
    timer = Timer()


    data_dir = "NLP_Dataset/MLHomework_Toxicity/train_.csv"
    test_dir = "NLP_Dataset/MLHomework_Toxicity/test.csv"
    val_dir = "NLP_Dataset/MLHomework_Toxicity/val_.csv"
    import pandas as pd

    df_test = pd.read_csv(test_dir)
    test_index = df_test['id']
    test_index = test_index.tolist()

    batch_size = 64

    max_seq_len = 500  #最大评论长度 单位为单词
    lr, num_epochs = 0.001, 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_train, target_train, index_train = read_comments(data_dir, True)
    data_test, target_test, index_test = read_comments(test_dir, False)
    data_val, target_val, index_val = read_comments(val_dir, True)
    train_tokens = tokenize(data_train, token='word')
    test_tokens = tokenize(data_test, token='word')
    val_tokens = tokenize(data_val, token='word')
    vocab = Vocab(train_tokens, min_freq=2, reserved_tokens=['<pad>'])

    train_features = torch.tensor([
        truncate_pad(vocab[line], max_seq_len, vocab['<pad>'])
        for line in train_tokens])

    test_features = torch.tensor([
        truncate_pad(vocab[line], max_seq_len, vocab['<pad>'])
        for line in test_tokens])
    val_features = torch.tensor([
        truncate_pad(vocab[line], max_seq_len, vocab['<pad>'])
        for line in val_tokens])

    dataset_train = CommentDataset(train_features, target_train, index_train)
    dataset_test = CommentDataset(test_features, None, index_test)
    dataset_val = CommentDataset(val_features, target_val, index_val)
    '''
        下面注释的是我关于非均匀采样的测试，因为数据集本身不是平衡的，正常评论远多于恶意的，
        所以设定了一个采样的权重
        然而实际效果并没有什么卵用，发现性能没有任何提升。。。
    '''
    # weights = []
    # for idx, (c, t, i) in enumerate(dataset_train):
    #     weights.append(t)
    #
    # train_sampler = WeightedRandomSampler(weights, len(weights))
    data_loaders = {
        #'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=False, sampler=train_sampler),
        'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True),
        'test': DataLoader(dataset_test, batch_size=batch_size, shuffle=False),
        "val": DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=True)
    }

    timer.query("Loading Data")


    '''
    主要有两个模型 ： BiRNN 和 TextCNN
    经过我的测试发现后者预测准确度高一点点
    '''
    # embed_size, num_hiddens, num_layers = 100, 100, 2
    # net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)



    '''
    权重初始化
    '''
    def init_weights(m):
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
        # if type(m) == torch.nn.LSTM:
        #     for param in m._flat_weights_names:
        #         if "weight" in param:
        #             torch.nn.init.xavier_uniform_(m._parameters[param])

    net.apply(init_weights)

    net = net.to(device)


    '''
    加载词嵌入参数
    可以理解为把单词转换为一个固定长度的向量，
    这里在训练的时候不需要更新embedding层的参数，所以将requires_grad置为false
    这里我们加载的是GLoVe模型的预训练参数，100维的词向量
    我试了试用300d的参数，但效果没啥区别。。。
    '''
    glove_embedding = TokenEmbedding('glove.6B.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    '''
    关于这个loss，我想了想还是用二分类交叉熵损失比较合适。
    之前用的是均方误差，但那样模型train不出来。
    '''
    loss = torch.nn.BCELoss()


    train(model=net, data_loaders=data_loaders, optimizer=opt, num_epochs=num_epochs,
          loss=loss, device=device)
    timer.query("Training")

    predict(net, data_loaders['test'])
    timer.query("Prediction")
