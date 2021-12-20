import datetime

from torch.utils.data import DataLoader, Dataset

from model import TextCNN,BiRNN
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


def read_comments(data_dir, is_train):
    import pandas as pd

    df = pd.read_csv(data_dir)

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
            if _i % 1000 == 0:
                print(f"training loss :   {epoch_loss}")
                validate(model, data_loaders['val'], loss, device)
        print("New Epoch Begin -------------------------------------")
        torch.save(model, 'model.pt')


def validate(model: torch.nn.Module, val_loader, loss, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, Y, index) in enumerate(val_loader):
            Y = Y.float().to(device)

            Y_hat = model(X)

            l = loss(Y_hat, Y)
            print(f"validating loss :   {l.sum().item()}")


def predict(model, test_loader):
    model.eval()

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

    data_dir = "NLP_Dataset/MLHomework_Toxicity/train2.csv"
    test_dir = "NLP_Dataset/MLHomework_Toxicity/test.csv"
    val_dir = "NLP_Dataset/MLHomework_Toxicity/val.csv"
    import pandas as pd

    df_test = pd.read_csv(test_dir)
    test_index = df_test['id']
    test_index = test_index.tolist()

    batch_size = 100
    max_seq_len = 500
    lr, num_epochs = 0.001, 10
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
    # print(dataset_train[0])
    data_loaders = {'train': DataLoader(dataset_train, batch_size=batch_size, shuffle=True),
                    'test': DataLoader(dataset_test, batch_size=batch_size, shuffle=False),
                    "val": DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=True)
                    }

    timer.query("Loading Data")

    # embed_size, num_hiddens, num_layers = 100, 100, 2
    # net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    embed_size, kernel_sizes, nums_channels = 300, [3, 4, 5], [300, 300, 300]
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)


    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == torch.nn.LSTM:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])


    # net.apply(init_weights)
    net = net.to(device)
    glove_embedding = TokenEmbedding('glove.42b.300d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.BCELoss()

    train(model=net, data_loaders=data_loaders, optimizer=opt, num_epochs=num_epochs,
          loss=loss, device=device)
    timer.query("Training")

    predict(net, data_loaders['test'])
    timer.query("Prediction")
