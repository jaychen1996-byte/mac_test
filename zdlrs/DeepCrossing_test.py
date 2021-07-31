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

# 使用具名元组定义特征标记
SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])


# 数据集
class Criteo(data.Dataset):
    def __init__(self, dense_features, sparse_features, labels):
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dense_features = self.dense_features.to_numpy()[idx]
        sparse_features = self.sparse_features.to_numpy()[idx]
        labels = self.labels.to_numpy()[idx]

        outs = [dense_features, sparse_features, labels]
        return outs


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
class DeepCrossing(nn.Module):
    def __init__(self,
                 embedding_classes,
                 residual_block_num=3,
                 embedding_dim=4,
                 sparse_classes=26,
                 dense_classes=13,
                 hidden_unit=256):
        super(DeepCrossing, self).__init__()
        self.residual_block_num = residual_block_num
        self.embedding = nn.ModuleList([nn.Embedding(ec + 1, embedding_dim) for ec in embedding_classes])
        self.all_features_cat = embedding_dim * sparse_classes + dense_classes
        self.residual_block = nn.ModuleList(
            [Residual_block(self.all_features_cat, hidden_unit) for _ in range(self.residual_block_num)])
        self.last_linear = nn.Linear(self.all_features_cat, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        dense_feature, sparse_feature, label = x[:]
        batch_features = None

        # 处理稀疏特征值
        sparse_features = []
        for sparse in sparse_feature:
            sfc = None
            for s, d in zip(sparse, self.embedding):
                out = d(s)
                if sfc == None:
                    sfc = out.unsqueeze(0)
                else:
                    sfc = torch.cat((sfc, out.unsqueeze(0)), 0)
            sparse_features.append(sfc.flatten())

        # 处理连续型特征(进行拼接)
        for df, sf in zip(dense_feature, sparse_features):
            if batch_features == None:
                batch_features = torch.cat((sf, df), 0).unsqueeze(0)
            else:
                batch_features = torch.cat((batch_features, torch.cat((sf, df), 0).unsqueeze(0)), 0)

        # 类型转换
        infer_data = batch_features.float()

        # forward
        for rb in self.residual_block:
            infer_data = rb(infer_data)
        out = self.last_linear(infer_data)
        out = self.sigmoid(out)

        return {"predicts": out, "labels": label}


# 获取到Embedding层的类别数
def cret_dataset_get_classes(data_root="../data/criteo_sample.txt", batch_size=4, shuffle=True, num_workers=0):
    # 读取数据
    data_df = pd.read_csv(data_root)
    # 划分dense和sparse特征
    columns = data_df.columns.values
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    # 将特征做标记
    dnn_feature_columns = [data_df[feat].nunique() for feat in sparse_features]

    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    for f in sparse_features:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])

    train_dataset = Criteo(data_df[dense_features], data_df[sparse_features], data_df["label"])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, dnn_feature_columns, len(dense_features), len(sparse_features)


def train(config):
    train_loader, embedding_classes, df_nums, sf_nums = cret_dataset_get_classes(config.data_root, config.batch_size)
    # 初始化模型
    model = DeepCrossing(embedding_classes,
                         config.residual_block_num,
                         config.embedding_dim,
                         sparse_classes=sf_nums,
                         dense_classes=df_nums,
                         hidden_unit=config.hidden_unit)
    # 初始化损失函数
    loss_fn = nn.BCELoss()
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epoch = range(config.epoch)

    with tqdm.tqdm(
            iterable=epoch,
            bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}'
    ) as t:
        for epc in epoch:
            start_time = datetime.now()
            losses = 0
            t.set_description_str(f"\33[36m【Epoch {epc + 1:04d}】")
            for batch in train_loader:
                out = model(batch)
                loss = loss_fn(out["predicts"].squeeze(1), out["labels"].float())
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
            cur_time = datetime.now()
            delta_time = cur_time - start_time
            t.set_postfix_str(f"epoch_loss={losses:.7f}， 执行时长：{delta_time}\33[0m")
            t.update()


def test():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_root',
                        default="../data/criteo_sample.txt",
                        type=str,
                        help='an integer for the accumulator')
    parser.add_argument('--batch_size',
                        default=4,
                        type=int,
                        help='an integer for the accumulator')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='nothing')
    parser.add_argument('--epoch',
                        default=300,
                        type=int,
                        help='nothing')
    parser.add_argument('--residual_block_num',
                        default=3,
                        type=int,
                        help='nothing')
    parser.add_argument('--embedding_dim',
                        default=4,
                        type=int,
                        help='nothing')
    parser.add_argument('--hidden_unit',
                        default=256,
                        type=int,
                        help='nothing')
    config = parser.parse_args()
    train(config)
