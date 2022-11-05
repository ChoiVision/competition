import albumentations as A
from albumentations.pytorch import ToTensorV2 

def train_aug(image_size):
  tr_aug = A.Compose([
                      A.Resize(image_size, image_size),
                      A.VerticalFlip(p=0.5),
                      A.HorizontalFlip(p=0.5),
                      A.ColorJitter(0.5),
                      A.ShiftScaleRotate(scale_limit=(0.7, 0.9), p=0.5, rotate_limit=30),
                      A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
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