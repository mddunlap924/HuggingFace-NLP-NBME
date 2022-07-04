from abc import ABC
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
from typing import Optional
from helper_fns import scoring

""" 
Helpful Links about Tokenizer 
# https://huggingface.co/docs/transformers/main_classes/tokenizer
"""


def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def create_label(cfg, text, annotation_length, location_list):
    encoded = cfg.tokenizer(text,
                            add_special_tokens=True,
                            max_length=cfg.max_len,
                            padding="max_length",
                            return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -1
    if annotation_length != 0:
        for location in location_list:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    "Takeaway - this returns tokenized/encoded 512 vector where 'label' identifies the location of the features in the tokenized words (-1=ignore b/c special token, 1=label, 0=no label"
    return torch.tensor(label, dtype=torch.float)


class NBMEDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.pn_historys[item],
                               self.feature_texts[item])
        label = create_label(self.cfg,
                             self.pn_historys[item],
                             self.annotation_lengths[item],
                             self.locations[item])
        return inputs, label, item


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.pn_historys[item],
                               self.feature_texts[item])
        return inputs


class NBMEDataModule(pl.LightningDataModule, ABC):
    def __init__(self, cfg: object, train: pd.DataFrame, val: pd.DataFrame):
        super().__init__()
        self.cfg = cfg
        self.train = train
        self.val = val
        self.val_dataset = None
        self.train_dataset = None
        self.train_texts = train['pn_history'].values
        self.train_labels = scoring.create_labels_for_scoring(train)
        self.val_texts = val['pn_history'].values
        self.val_labels = scoring.create_labels_for_scoring(val)

    def setup(self, stage: Optional[str] = None):
        # Setup Training Dataset
        self.train_dataset = NBMEDataset(self.cfg, self.train)

        # Setup Validation Dataset
        self.val_dataset = NBMEDataset(self.cfg, self.val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.num_workers,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.batch_size,
                          shuffle=False,
                          num_workers=self.cfg.num_workers,
                          pin_memory=True,
                          )
