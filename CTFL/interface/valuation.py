import numpy as np
from dataprep.dataprep import DataFed, get_data_loader
from train import train_model, test_model
from utils.args import args


class Valuation(object):
    def __init__(self, datafed: DataFed, db_enc):
        self.dfed = datafed
        self.db_enc = db_enc
        y = self.dfed.te[1]
        if len(self.dfed.te[1].size()) > 1: y = y.argmax(dim=-1)
        self.random_score = np.max(np.bincount(y)) / y.size(0)

    def __call__(self, pidxs, test_bagging=False):
        if not len(pidxs): return self.random_score
        data_tr = self.dfed.comb_tr(pidxs)
        data_te = self.dfed.te if not test_bagging else self.dfed.sample_te()
        train_loader, valid_loader, test_loader = get_data_loader(data_tr, data_te, args.batch_size,
                                                                  pin_memory=True, save_best=args.save_best)
        train_model(self.db_enc, train_loader, valid_loader)
        acc, f1 = test_model(test_loader)
        return acc
