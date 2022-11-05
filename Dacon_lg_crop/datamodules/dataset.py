import cv2
from torch.utils.data import Dataset

class TrainDataset(Dataset):
  def __init__(self, df, augmentation = None):
    self.df = df
    self.augmentation = augmentation

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    img_path = self.df.image_path[index]
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    label = self.df.label_unique[index]

    if self.augmentation is not None:
      img = self.augmentation(image = img)['image']

    return img, label

class TestDataset(Dataset):
  def __init__(self, df, augmentation = None):
    self.df = df
    self.augmentation = augmentation

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    img_path = self.df.image_path[index]
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    if self.augmentation is not None:
      img = self.augmentation(image = img)['image']

    return img

