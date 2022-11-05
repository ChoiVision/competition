import os
import json
import random
import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm

import torch


import pytorch_lightning as pl
from train.inference import test

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
    print('Log File Exist')

def create_dataframe(args):

  train_csv = sorted(glob(args.train_csv))
  train_jpg = sorted(glob(args.train_image))
  train_json = sorted(glob(args.train_json))
  test_jpg = sorted(glob(args.test_image))

  crops = []
  diseases = []
  risks = []
  labels = []

  for i in tqdm(range(len(train_json)),desc='Read Json'):
    with open(train_json[i], 'r') as f:
      sample = json.load(f)
      crop = sample['annotations']['crop']
      disease = sample['annotations']['disease']
      risk = sample['annotations']['risk']
      label=f"{crop}_{disease}_{risk}"
  
      crops.append(crop)
      diseases.append(disease)
      risks.append(risk)
      labels.append(label)

  train_df = pd.DataFrame({'image_path': train_jpg,'label':labels})

  label_index = pd.factorize(train_df['label'])[1]
  train_df['label_unique']= pd.factorize(train_df['label'])[0]
  
  test_df = pd.DataFrame({'image_path': test_jpg})

  train_df.to_csv(args.train_df, index=False)
  test_df.to_csv(args.test_df, index=False)

  return label_index, train_df, test_df


def submission(args, label):
  sub = pd.read_csv(args.sub_file)
  sub['label'] = label
  sub.to_csv(f'./submission/{args.model_name}.csv', index=False)