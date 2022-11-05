import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import torch
import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_image_path(id, dir):
  return f'{dir/id}'


def seed_everything(args):
    random.seed(args)
    np.random.seed(args)
    os.environ["PYTHONHASHSEED"] = str(args)
    torch.manual_seed(args)
    torch.cuda.manual_seed(args) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True  
    pl.seed_everything(args)

def create_log_dir(args):
  if not os.path.exists(args):
    os.mkdir(args)

def preprocessing_df(args, mode = 'train'):
  if mode == 'train':
    df = pd.read_csv(f'{args.default_dir}train.csv')
    encoder = LabelEncoder()
    df['individual_id'] = encoder.fit_transform(df['individual_id'])
    np.save(args.default_dir + 'encoder_classes.npy', encoder.classes_)
    df['image_path'] = args.train_image_dir + df['image']
    df.drop(['species'], axis=1, inplace=True)


  elif mode == 'test':
    df = pd.read_csv(f'{args.default_dir}sample_submission.csv')
    df.drop(['predictions'], axis=1 , inplace=True)
    df['image_path'] = args.test_image_dir + df['image']
    df['individual_id'] = 0
  
  df.to_csv(f'{args.default_dir}{mode}.csv', index=False)

  return df

def train_aug(image_size):
  tr_aug = A.Compose([
                    A.Resize(image_size, image_size),
                    A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.25), shear=(-3, 3), p=0.5),
                    A.ToGray(p=0.1),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.05),
                    A.GaussNoise(p=0.05),
                    A.RandomGridShuffle(grid=(2, 2), p=0.3),
                    A.Posterize(p=0.2),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Cutout(p=0.05),
                    A.RandomSnow(p=0.1),
                    A.RandomRain(p=0.05),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(p=1),
                    ToTensorV2(p=1)
                      ])


  return tr_aug

def test_aug(image_size):
  ts_aug = A.Compose([
                      A.Resize(image_size, image_size),
                      A.VerticalFlip(p=0.5),
                      A.HorizontalFlip(p=0.5),
                      A.Normalize(p=1),
                      ToTensorV2(p=1)
                      ])

  return ts_aug