import os
import random
import pandas as pd
import numpy as np

import torch

import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2 

from train.eval import test

def seed_everything(args):
    random.seed(args)
    np.random.seed(args)
    os.environ["PYTHONHASHSEED"] = str(args)
    torch.manual_seed(args)
    torch.cuda.manual_seed(args) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True  
    pl.seed_everything(args)

def create_log_dir(log):
  if not os.path.exists(log):
    print('Create Log File')
    os.mkdir(log)
  else:
    pass

def create_df(args):
    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)

    train['img_path'] = os.getcwd() +  '/' + train['img_path']
    test['img_path'] = os.getcwd() + '/' + test['img_path']
    
    train.to_csv('./train.csv', index=False)
    test.to_csv('./test.csv', index=False)


def train_aug(image_size):
  tr_aug = A.Compose([
                      A.Resize(image_size, image_size),
                      A.VerticalFlip(p=0.5),
                      A.HorizontalFlip(p=0.5),
                      A.ShiftScaleRotate(scale_limit=(0.7, 0.9), p=0.5, rotate_limit=30),
                      A.Blur(p=0.3),
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



def submission(args, label):
  sub = pd.read_csv(args.sub_file)
  sub['label'] = label
  sub.to_csv(f'{args.submission}{args.model_name}.csv', index=False)