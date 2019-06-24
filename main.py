import git
import os
import argparse
import comet_ml
import torch

from torch.optim import lr_scheduler, Adam
from data   import load_LS3D as load_data
from models import HRFPN34 as Model
from utils  import Trainer, Logger, LR_Finder

parser = argparse.ArgumentParser()
parser.add_argument('--train-root'  , type=str, required=True)
parser.add_argument('--val-root'    , type=str, required=True)
parser.add_argument('--log-root'    , type=str, default='~/Experiments/')
parser.add_argument('--image-size'  , type=int, default=512)
parser.add_argument('--pretrained'  , type=str, default='')
parser.add_argument('--batch-size'  , type=int, default=32)
parser.add_argument('--epoch'       , type=int, default=300)
parser.add_argument('--num-workers' , type=int, default=8)
parser.add_argument('--lr'          , type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--lr-patience' , type=int, default=150)
parser.add_argument('--cuda'        , action='store_true', default=False)
parser.add_argument('--find-lr'     , action='store_true', default=False)
parser.add_argument('--track'       , action='store_true', default=False)
parser.add_argument('--comment'     , type=str, default='')
args = parser.parse_args()

torch.manual_seed(0)
if args.cuda:
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark = args.track

# Setup Dataloader
data = load_data(
    args.train_root, args.val_root,
    args.image_size, args.batch_size,
    args.num_workers, args.cuda
)

# Setup Model
model = Model(68 + 1)

# Setup Optimizer
decay_params = []
no_decay_params = []
for k, p in model.named_parameters():
    if 'bias' in k or 'bn' in k:
        no_decay_params.append(p)
    else:
        decay_params.append(p)

model_optimizer = Adam([{'params': no_decay_params},
                        {'params': decay_params, 'weight_decay': args.weight_decay}],
                       lr=args.lr)

if args.pretrained:
    data = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(data['state_dict'])
    model_optimizer.load_state_dict(data['optim'])

if args.cuda:
    model = model.cuda()
    model = torch.nn.DataParallel(model)

# Setup Logger
dirname   = os.path.dirname(os.path.realpath(__file__))
repo      = git.repo.Repo(dirname)

if args.find_lr:
    lr_finder = LR_Finder(
        data, model, model_optimizer, args.cuda, name='model'
    )
    lr_finder.run()
    exit(0)

if args.track and repo.is_dirty():
    print("Commit before running trackable experiments")
    exit(-1)

commit_id = repo.commit().hexsha
logger    = Logger(
    args.log_root, 'face-keypoint',
    commit_id, comment=args.comment,
    disabled=(not args.track)
)

scheduler = lr_scheduler.StepLR(
    model_optimizer, args.lr_patience, gamma=0.1
)

# Setup and start Trainer
trainer = Trainer(
    data, model, scheduler, model_optimizer,
    logger, args.cuda
)

trainer.run(args.epoch)
