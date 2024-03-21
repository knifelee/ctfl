import torch
from torch.utils.tensorboard import SummaryWriter
from ours.implementation import RuleModel
from utils.args import args


def train_model(db_enc, train_loader, valid_loader=None):
    writer = SummaryWriter(args.folder_path)
    y_fname = db_enc.y_fname
    discrete_flen = db_enc.discrete_flen
    continuous_flen = db_enc.continuous_flen

    model = RuleModel(dim_list=[(discrete_flen, continuous_flen)] + list(map(int, args.structure.split('@'))) + [len(y_fname)],
                    device_id=args.device,
                    use_not=args.use_not,
                    is_rank0=True,
                    log_file=args.log,
                    writer=writer,
                    estimated_grad=args.estimated_grad,
                    save_path=args.model)

    model.train_model(
        data_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.learning_rate,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay,
        log_iter=args.log_iter)
    return model


def load_model(path, device_id, log_file=None):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['model_args']
    model = RuleModel(
        dim_list=saved_args['dim_list'],
        device_id=device_id,
        is_rank0=True,
        use_not=saved_args['use_not'],
        log_file=log_file,
        estimated_grad=saved_args['estimated_grad'])
    stat_dict = checkpoint['model_state_dict']
    for key in list(stat_dict.keys()):
        if 'module' in key: stat_dict[key[7:]] = stat_dict.pop(key)
    model.net.load_state_dict(checkpoint['model_state_dict'])
    return model


def test_model(test_loader):
    model = load_model(args.model, args.device_ids[0], log_file=args.test_res)
    _, acc, _, f1 = model.test(test_loader=test_loader, set_name='Test')
    return acc, f1
