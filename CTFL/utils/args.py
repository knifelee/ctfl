import os
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument('-i', '--device_ids', type=str, default='3', help='Set the device (GPU ids). Split by @.'
                                                                      ' E.g., 0@1')
parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
parser.add_argument('-e', '--epoch', type=int, default=401, help='Set the total epoch.')
parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Set the batch size.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='Set the initial learning rate.')
parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.75, help='Set the learning rate decay rate.')
parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=200, help='Set the learning rate decay epoch.')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6, help='Set the weight decay (L2 penalty).')
parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th 5-fold validation, 0 <= ki < 5.')
parser.add_argument('-rc', '--round_count', type=int, default=0, help='Count the round of experiments.')
parser.add_argument('-ma', '--master_address', type=str, default='127.0.0.1', help='Set the master address.')
parser.add_argument('-mp', '--master_port', type=str, default='12345', help='Set the master port.')
parser.add_argument('-li', '--log_iter', type=int, default=500, help='The number of iterations (batches) to log once.')

parser.add_argument('--use_not', action="store_true",
                    help='Use the NOT (~) operator in logical rules. '
                         'It will enhance model capability but make the model more complex.')
parser.add_argument('--save_best', action="store_true",
                    help='Save the model with best performance on the validation set.')
parser.add_argument('--estimated_grad', action="store_true",
                    help='Use estimated gradient.')
parser.add_argument('-s', '--structure', type=str, default='1@16',
                    help='Set the number of nodes in the binarization layer and logical layers. '
                         'E.g., 10@64, 10@64@32@16.')

parser.add_argument('-m', '--method', type=str, default='CTFL', help='Evaluation method.')
parser.add_argument('-se', '--seed', type=int, default=60, help='Random seed.')
parser.add_argument('-pd', '--pdist', type=str, default='skew_label', help='Data distribution of participants.')
parser.add_argument('-a', '--alpha', type=float, default=1, help='Data distribution skewness')

try:
    args = parser.parse_args()
except:
    args = parser.parse_args(args=[])

args.folder_name = '{}_{}_{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_ki{}_rc{}_useNOT{}_saveBest{}_estimatedGrad{}'.format(
    args.method, args.pdist, args.data_set, args.epoch, args.batch_size, args.learning_rate, args.lr_decay_rate,
    args.lr_decay_epoch, args.weight_decay, args.ith_kfold, args.round_count, args.use_not,
    args.save_best, args.estimated_grad)

args.folder_name = args.folder_name + '_L' + args.structure
args.set_folder_path = os.path.join('logs', args.data_set)
if not os.path.exists(args.set_folder_path):
    os.mkdir(args.set_folder_path)
args.folder_path = os.path.join(args.set_folder_path, args.folder_name)
if not os.path.exists(args.folder_path):
    os.mkdir(args.folder_path)
args.model = os.path.join(args.folder_path, 'model.pth')
args.model_file = os.path.join(args.folder_path, 'ours.txt')
args.plot_file = os.path.join(args.folder_path, 'plot_file.pdf')
args.log = os.path.join(args.folder_path, 'log.txt')
args.test_res = os.path.join(args.folder_path, 'test_res.txt')
args.device_ids = list(map(lambda x: int(x) % 4, args.device_ids.strip().split('@')))
args.gpus = len(args.device_ids)
# args.device = ('cuda:{}'.format(args.device_ids[0] % 4) if torch.cuda.is_available() else 'cpu')
args.device = ('cpu')
args.nodes = 1
args.world_size = args.gpus * args.nodes
args.batch_size = int(args.batch_size / args.gpus)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
