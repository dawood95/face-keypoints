import inspect
import shutil
from pathlib import Path

import torch
from tensorboardX import SummaryWriter

class Logger:

    def __init__(self, log_root, exp_name, comments):
        log_root = Path(log_root).expanduser()
        if not log_root.is_dir():
            log_root.mkdir(0o755)

        exp_num = len(list(log_root.rglob(exp_name+'*'))) + 1
        exp_name += '_'+str(exp_num)

        log_dir = log_root / exp_name
        writer  = SummaryWriter(log_dir=log_dir.as_posix(), comment=comments)

        self.exp_name = exp_name
        self.log_dir = log_dir
        self.writer = writer

    def cache_model(self):
        proj_dir = Path(inspect.getfile(Logger)).parent.parent
        shutil.make_archive((self.log_dir / 'pose-estimation').as_posix(), 'zip', proj_dir.as_posix())
        print("Archiving working directory at %s"%self.log_dir.as_posix())

    def save(self, name, data):
        torch.save(data, (self.log_dir / name).as_posix())
