import time
import argparse
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch import nn
from torch import optim

from models.openpose import Openpose as Model

from data import get_COCO as getData
from trainer import Trainer
from utils.logger import Logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log-root', type=str, default='~/Experiments/')
    parser.add_argument('--comments', type=str, default='')
    return parser.parse_args()

args = get_args()


def main():

    # Logging
    exp_name  = 'HM+JM_'+Model.__qualname__
    logger = Logger(args.log_root, exp_name, args.comments)
    logger.writer.add_text('args', str(args))

    # Data
    data = getData(args.data_root, args.batch_size, args.num_workers, args.cuda)
    train_loader, val_loader, (cocoGt, imgIds) = data

    # Model
    model = Model()
    optimizer = optim.Adam(params = model.parameters(), lr = args.lr, amsgrad=True)

    model_str = str(model).replace(' ', '&nbsp;').replace('\n', '<br>')
    logger.writer.add_text('model_def', '''%s'''%(model_str))
    logger.cache_model()

    if args.cuda: model = model.cuda()
    if args.pretrained: model.load_state_dict(torch.load(args.pretrained))
    if args.cuda: model = torch.nn.DataParallel(model)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader,
                      cocoGt, imgIds,
                      optimizer, logger, args.cuda)
    trainer.run(args.epoch)

if __name__ == "__main__": main()



