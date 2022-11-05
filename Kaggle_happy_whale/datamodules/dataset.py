import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TrainDataset(Dataset):
  def __init__(self, df, transform):
    self.df= df
    self.transform= transform
    self.image_names= self.df['image'].values
    self.image_paths= self.df['image_path'].values
    self.target= self.df['individual_id'].values

  def __getitem__(self, index):
    image_name= self.image_names[index]
    image_path= self.image_paths[index]
    image= cv2.imread(image_path)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label= self.target[index]

    if self.transform is not None:
      image= self.transform(image= image)['image']
    
    return {
            'image_name': image_name,
            'image': image,
            'target': label
            }
    
  def __len__(self):
    return len(self.df)

class TestDataset(Dataset):
  def __init__(self, df, transform):
    self.df= df
    self.transform= transform
    self.image_names= self.df['image'].values
    self.image_paths= self.df['image_path'].values
    self.target= self.df['individual_id'].values

  def __getitem__(self, index):
    image_name= self.image_names[index]
    image_path= self.image_paths[index]
    image= cv2.imread(image_path)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.transform is not None:
      image= self.transform(image= image)['image']
    
    return {
            'image_name': image_name,
            'image': image
            }
    
  def __len__(self):
    return len(self.df)