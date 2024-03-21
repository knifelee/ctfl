import os
from math import floor, ceil
import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from utils import config
from scipy.stats import dirichlet
from utils.args import args


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    D = pd.read_csv(data_path, header=None)
    if shuffle: D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    D.columns = f_df.iloc[:, 0]
    y_df = D.iloc[:, [label_pos]]
    X_df = D.drop(D.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return X_df, y_df, f_df, label_pos


def read_data(dataset):
    data_path = os.path.join(config.DATA_DIR, dataset + '.data')
    info_path = os.path.join(config.DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)
    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)
    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)
    X, y = torch.FloatTensor(X), torch.FloatTensor(y)
    split_pos = int(len(X) * config.SPLIT)
    return db_enc, (X[split_pos:], y[split_pos:]), (X[:split_pos], y[:split_pos])


def get_data_loader(tr, te, batch_size, pin_memory=False, save_best=True):
    train_set = TensorDataset(*tr)
    test_set = TensorDataset(*te)

    train_len = int(len(train_set) * 0.95)
    train_sub, valid_sub = random_split(train_set, [train_len, len(train_set) - train_len])
    if not save_best:  # use all the training set for training, and no validation set used for model selections.
        train_sub = train_set

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_sub, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df.values)
        self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data.values)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names_out(feature_names))
            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns
            self.discrete_flen = 0
        self.continuous_flen = continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = self.feature_enc.transform(discrete_data)
            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                X_df = pd.DataFrame(discrete_data.toarray())
        else:
            X_df = continuous_data
        return X_df.values, y


class DataFed():
    def __init__(self, num_parts, data_tr: (torch.Tensor, torch.Tensor), data_te: (torch.Tensor, torch.Tensor),
                 split_type=None, ratios=None):
        self.nparts = num_parts
        self.split_type = split_type
        self.ratios = [1. / self.nparts] * self.nparts if not ratios else ratios
        if split_type == 'skew_sample':
            self.idxs, (self.Xs, self.Ys) = self.skew_sample_split(data_tr)
        else:
            self.idxs, (self.Xs, self.Ys) = self.skew_label_split(data_tr)
        self.tr = self.comb_tr(list(range(num_parts)))
        self.te = data_te
        self.tr_loader = DataLoader(TensorDataset(*self.tr), batch_size=config.BATCH_SIZE, pin_memory=True)
        self.te_loader = DataLoader(TensorDataset(*self.te), batch_size=config.BATCH_SIZE, pin_memory=True)
        # print(f'training data split sizes:{len(self.tr[0])} ,{[len(idx) for idx in self.idxs]}')

    def comb_tr(self, pidxs):
        return tuple(torch.cat(tuple(D[i] for i in pidxs)) for D in (self.Xs, self.Ys))

    def sample_te(self):  # for bagging a average accu
        X, Y = self.te
        idxs = torch.ones(len(X)).multinomial(num_samples=len(X), replacement=True)
        return X[idxs], Y[idxs]

    def skew_sample_split(self, tr):
        ports = dirichlet.rvs([args.alpha] * self.nparts, random_state=config.SEED)[0]
        idxs = [list(range(int(sum(ports[:i]) * len(tr[0])), min(len(tr[0]), int(
            sum(ports[:i + 1]) * len(tr[0]))))) for i in range(self.nparts)]
        return idxs, ([X[idx] for idx in idxs] for X in tr)

    def skew_label_split(self, tr):  # fix #samples, varying negative from 0% - 100%
        lspace = tr[1].size()[-1]
        ports = dirichlet.rvs([args.alpha] * self.nparts, size=lspace, random_state=config.SEED)
        idxs_y = [(tr[1].argmax(-1) == i).nonzero().flatten() for i in range(lspace)]
        idxs = [torch.cat([idxs_y[l][floor(sum(ports[l][:cidx]) * len(idxs_y[l])): ceil(
            sum(ports[l][:cidx + 1]) * len(idxs_y[l]))] for l in range(lspace)]) for cidx in range(self.nparts)]
        Xs, Ys = ([X[idx] for idx in idxs] for X in tr)
        return [idx.tolist() for idx in idxs], (Xs, Ys)
