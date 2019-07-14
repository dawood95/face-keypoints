from comet_ml import Experiment
from pathlib import Path
import git
import torch


class Logger:

    def __init__(self, log_dir, project_name, commit_id,
                 comment=None, disabled=True):

        # setup comet-ml
        key_path = Path('~/.cometml').expanduser().as_posix()
        api_key = open(key_path).read().strip()
        experiment = Experiment(api_key, project_name,
                                disabled=disabled)
        
        experiment.log_parameter('commit_id', commit_id)
        if comment:
            experiment.log_other('comment', comment)

        # setup model backup dir
        exp_name = project_name + str(experiment.id)
        log_dir = Path(log_dir).expanduser() / exp_name
        if not log_dir.is_dir() and not disabled:
            log_dir.mkdir(0o755)

        self.log_dir  = log_dir
        self.comet    = experiment
        self.disabled = disabled

    def save(self, name, data):
        if self.disabled: return
        torch.save(data, (self.log_dir / name).as_posix())
