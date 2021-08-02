import torch
import tqdm
import numpy as np
import pandas as pd
from torch import nn
from torch.utils import data
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder
import argparse
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

# 使用具名元组定义特征标记
SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])


# 数据集
class MovieLens(data.Dataset):
    def __init__(self, train_datas):
        self.train_datas = train_datas

    def __len__(self):
        return len(self.train_datas)

    def __getitem__(self, idx):
        return self.train_datas[idx]


# 残差模块
class Residual_block(nn.Module):
    def __init__(self, dim_stack, hidden_unit):
        super(Residual_block, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, dim_stack)
        self.relu = nn.ReLU()

    def forward(self, x):
        orig_x = x.clone()
        x = self.linear1(x)
        x = self.linear2(x)
        out = self.relu(x + orig_x)
        return out


# 模型架构
class NCF(nn.Module):
    def __init__(self,
                 embedding_classes,
                 embedding_dim=8,
                 hidden_unit=32):
        super(NCF, self).__init__()
        self.GMF_embedding = nn.ModuleList([nn.Embedding(ec + 1, embedding_dim) for ec in embedding_classes])
        self.MLP_embedding = nn.ModuleList([nn.Embedding(ec + 1, embedding_dim) for ec in embedding_classes])
        self.all_features_cat = embedding_dim * 2
        self.linear1 = nn.Linear(self.all_features_cat, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, self.all_features_cat)
        self.last_linear = nn.Linear(self.all_features_cat + embedding_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        user_feature, movie_feature, rating = x[:, 0], x[:, 1], x[:, 2]

        GMF = torch.mul(self.GMF_embedding[0](user_feature), self.GMF_embedding[1](movie_feature))

        MLP = torch.cat((self.MLP_embedding[0](user_feature), self.MLP_embedding[1](movie_feature)), 1)
        MLP = self.linear1(MLP)
        MLP = self.linear2(MLP)
        MLP = self.relu(MLP)

        NeuMF = torch.cat((GMF, MLP), 1)
        out = self.last_linear(NeuMF)

        return {"predicts": out, "labels": rating}


# 获取到Embedding层的类别数
def cret_dataset_get_classes(data_root="../data/ml-1m/ratings.dat", batch_size=1, shuffle=True, num_workers=0):
    # 读取数据，NCF使用的特征只有user_id和item_id
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    data_df = pd.read_csv(data_root, sep='::', engine="python", names=rnames)

    lbe = LabelEncoder()
    data_df['user_id'] = lbe.fit_transform(data_df['user_id'])
    data_df['movie_id'] = lbe.fit_transform(data_df['movie_id'])

    train_data = data_df[['user_id', 'movie_id']]
    train_data['label'] = data_df['rating']

    dnn_feature_columns = [train_data['user_id'].nunique(), train_data['movie_id'].nunique()]

    train_dataset = MovieLens(train_data.to_numpy())
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, dnn_feature_columns


def train(config):
    train_loader, embedding_classes = cret_dataset_get_classes(config.data_root, config.batch_size)

    # 初始化模型
    model = NCF(embedding_classes,
                config.embedding_dim,
                hidden_unit=config.hidden_unit)

    # print(model)

    # 初始化损失函数
    loss_fn = nn.MSELoss()
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epoch = range(config.epoch)

    for epc in epoch:
        with tqdm.tqdm(
                iterable=train_loader,
                bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}'
        ) as t:
            start_time = datetime.now()
            t.set_description_str(f"\33[36m【Epoch {epc + 1:04d}】")
            for batch in train_loader:
                out = model(batch)
                loss = loss_fn(out["predicts"].squeeze(1), out["labels"].float())
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cur_time = datetime.now()
                delta_time = cur_time - start_time
                t.set_postfix_str(f"train_loss={loss.item():.7f}， 执行时长：{delta_time}\33[0m")
                t.update()


def test():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_root',
                        default="../data/ml-1m/ratings.dat",
                        type=str,
                        help='an integer for the accumulator')
    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='an integer for the accumulator')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='nothing')
    parser.add_argument('--epoch',
                        default=2,
                        type=int,
                        help='nothing')
    parser.add_argument('--embedding_dim',
                        default=8,
                        type=int,
                        help='nothing')
    parser.add_argument('--hidden_unit',
                        default=64,
                        type=int,
                        help='nothing')
    config = parser.parse_args()
    train(config)
