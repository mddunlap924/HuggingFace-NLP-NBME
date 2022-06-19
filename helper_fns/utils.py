import os
import yaml
from helper_fns import scoring
import random
import numpy as np
import torch
import math
import time
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
import wandb


# class Dict2Class(object):
#     """" Turns a dictionary into a class """
#     def __init__(self, my_dict):
#         for key in my_dict:
#             setattr(self, key, my_dict[key])


def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]

        # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
        return d

    # declaring a class
    class C:
        pass

    # constructor of the class passed to obj
    obj = C()

    for k in d:
        obj.__dict__[k] = dict2obj(d[k])

    return obj


def load_cfg(filename):
    cfg_path = os.path.join(os.getcwd(), 'cfgs', filename)
    with open(cfg_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    file.close()
    # cfg = Dict2Class(my_dict=cfg_dict)
    cfg = dict2obj(d=cfg_dict)
    return cfg


def get_logger(*, filename='./results/' + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    seed_everything(seed)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


class WandB:
    """ Log results for Weights & Biases """

    def __init__(self, cfg, group_id):
        del cfg.checkpoint, cfg.tokenizer
        self.cfg = cfg
        self.group_id = group_id
        self.job_type = wandb.util.generate_id()
        self.version = wandb.util.generate_id()
        self.run_name = f'{self.group_id}_{self.job_type}'
        self.id = wandb.util.generate_id()  # Generate version name for tracking in wandb
        # WandB Key in the environment variables
        if 'WANDB_KEY' in os.environ:
            self.key = os.environ['WANDB_KEY']
        else:
            self.key = None

    def login_to_wandb(self):
        # Store WandB key in the environment variables
        if self.key is not None:
            wandb.login(key=self.key)
        else:
            print('Not logging info in WandB')

    def get_logger(self):
        self.login_to_wandb()
        wb_logger = pl_loggers.WandbLogger(project=self.cfg.wandb_project,
                                           group=self.group_id,
                                           job_type=self.job_type,
                                           name=self.run_name,
                                           version=self.version,
                                           config=self.cfg,
                                           )

        return wb_logger
