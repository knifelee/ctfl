from interface.valuation import Valuation
from dataprep.dataprep import DataFed, read_data
from utils.args import args
from utils import config
from ours.ctfl import CTFL



if __name__ == '__main__':
    db_enc, tr, te = read_data(args.data_set)
    v = Valuation(DataFed(config.NUM_PARTS, tr, te, 'skew_label'), db_enc)
    model = CTFL(v)
    model.run()
