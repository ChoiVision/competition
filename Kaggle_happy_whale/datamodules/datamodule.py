import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datamodules.dataset import TrainDataset, TestDataset
from utils.utils import train_aug, test_aug

class CustomModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.args= args
    self.train_df= pd.read_csv(f'{args.default_dir}train.csv')
    self.test_df= pd.read_csv(f'{args.default_dir}test.csv')
    self.split_kfold()
  
  def set_fold_num(self, fold_num):
    self.fold_num= fold_num

  def split_kfold(self):
    skf = StratifiedKFold(n_splits=self.args.folds, shuffle = True, random_state = self.args.seed)

    for fold, (_, val) in enumerate(skf.split(X = self.train_df, y = self.train_df['individual_id'])):
      self.train_df.loc[val, 'kfold'] = int(fold)    

  def setup(self, stage=None):
    if stage == 'fit' or stage is None:
      train_df= self.train_df[self.train_df['kfold'] != self.fold_num].reset_index(drop=True)
      valid_df= self.train_df[self.train_df['kfold'] == self.fold_num].reset_index(drop=True)

      self.train_ds= TrainDataset(train_df, transform=train_aug(self.args.image_size))
      self.val_ds= TrainDataset(train_df, transform=train_aug(self.args.image_size))

    if stage == 'test' or stage is None:
      self.test_ds= TestDataset(self.test_df, transform=test_aug(self.args.image_size))

  def train_dataloader(self):
    return self._dataloader(self.train_ds, train=True)

  def val_dataloader(self):
    return self._dataloader(self.val_ds)

  def test_dataloader(self):
    return self._dataloader(self.test_ds)

  def _dataloader(self, dataset, train=False):
    return DataLoader(
        dataset,
        batch_size= self.args.batch_size,
        shuffle= train,
        num_workers= self.args.num_workers,
        pin_memory= True,
    )