import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from datamodules.dataset import TrainDataset, TestDataset
from utils.trans import train_aug, test_aug

class CustomDataModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.train_df = pd.read_csv(os.path.join(args.train_df))
    self.test_df = pd.read_csv(os.path.join(args.test_df))
    self.split_kfold()
  
  def set_fold_num(self, fold_num):
    self.fold_num = fold_num
    
  def setup(self, stage=None):
    if stage == 'fit' or stage is None:
      print(f'FOLD:{self.fold_num}')

      train_df = self.train_df[self.train_df['kfold'] != self.fold_num].reset_index(drop=True)
      valid_df = self.train_df[self.train_df['kfold'] == self.fold_num].reset_index(drop=True)
      
      self.train = TrainDataset(train_df, augmentation = train_aug(self.args.image_size))
      self.valid = TrainDataset(valid_df, augmentation = train_aug(self.args.image_size))

    if stage == 'test' or stage is None:
      self.test = TestDataset(self.test_df, augmentation = test_aug(self.args.image_size))

  def split_kfold(self):
    skf = StratifiedKFold(n_splits=self.args.folds, shuffle = True, random_state = self.args.seed)

    for fold, (_, val) in enumerate(skf.split(X = self.train_df, y = self.train_df['label_unique'])):
      self.train_df.loc[val, 'kfold'] = int(fold)

  def train_dataloader(self):
    return self._dataloader(self.train)
    
  def val_dataloader(self):
    return self._dataloader(self.valid)

  def test_dataloader(self):
    return self._dataloader(self.test)

  def _dataloader(self, dataset, is_train=False):
    return DataLoader(dataset= dataset,
                      batch_size= self.args.batch_size,
                      num_workers= self.args.num_workers,
                      pin_memory= True,
                      shffule= is_train)

