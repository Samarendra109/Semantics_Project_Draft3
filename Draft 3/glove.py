import pickle
import warnings
from dataclasses import dataclass

import torch
from torch import nn, optim, sqrt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from util import get_data, word2index_file, get_glove_vectors_and_train_indices


@dataclass
class Config:
    y_max = 100
    alpha = 3 / 4
    embed_size = 25
    path = "glove_brown.pt"


class GloveModel(nn.Module):

    def __init__(self, word_embeddings, train_indices):
        super().__init__()
        self.w_center = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        self.w_contex = nn.Embedding.from_pretrained(word_embeddings, freeze=False)

        self.b_center = nn.Parameter(torch.zeros(word_embeddings.size(0)))
        self.b_contex = nn.Parameter(torch.zeros(word_embeddings.size(0)))

        # Will check this, I expect this to be of same size as word_embeddings.size(0) with boolean values
        self.train_indices = train_indices

    def forward(self, indices):
        center_indices = indices[:, 0]
        context_indices = indices[:, 1]
        return torch.einsum("bi,bi->b", self.w_center(center_indices), self.w_contex(context_indices)) + \
               self.b_center[center_indices] + self.b_contex[context_indices]

    def freeze_grad(self):
        self.w_center.weight.grad[~self.train_indices, :] = 0
        self.w_contex.weight.grad[~self.train_indices, :] = 0


def get_fy(y):
    fy = (y / Config.y_max)
    fy[fy > 1] = 1
    fy = fy ** Config.alpha

    return fy


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def train(X, y, epochs=1000):
    warnings.filterwarnings("ignore")
    model = GloveModel(*get_glove_vectors_and_train_indices())
    #model.load_state_dict(torch.load(Config.path))

    fy = get_fy(y)
    log_y = torch.log(1 + y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters())
    dataset = TensorDataset(X, fy, log_y)
    loader = DataLoader(dataset, batch_size=10_000, shuffle=True, num_workers=4)

    for t in range(epochs):
        running_loss = 0.0
        for X_i, fy_i, log_y_i in tqdm(loader):
            X_i = X_i.to(device)
            fy_i = fy_i.to(device)
            log_y_i = log_y_i.to(device)

            optimizer.zero_grad()
            y_pred = model(X_i)
            loss = weighted_mse_loss(log_y_i, y_pred, fy_i)
            loss.backward()
            model.freeze_grad()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {t}: {running_loss}")

    torch.save(model.state_dict(), Config.path)


def save_pkl_for_jupyter():
    with open(word2index_file, "rb") as f:
        word2index = pickle.load(f)

    model = GloveModel(*get_glove_vectors_and_train_indices())
    model.load_state_dict(torch.load(Config.path))

    word2vec = (model.w_center(torch.arange(len(word2index), dtype=torch.int64)) +
                model.w_contex(torch.arange(len(word2index), dtype=torch.int64))) / 2
    word2vec = word2vec.detach().numpy()

    with open("glove_brown_model.pkl", "wb") as f:
        pickle.dump((word2vec, word2index), f)


def tmp_test():
    X, y = get_data()
    model = GloveModel(*get_glove_vectors_and_train_indices())
    model.load_state_dict(torch.load(Config.path))
    y_pred = model(X)
    log_y = torch.log(y)

    print("Hello")


if __name__ == "__main__":
    X, y = get_data()
    train(X, y)
    save_pkl_for_jupyter()
    # tmp_test()
