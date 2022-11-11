import albumentations as A
from albumentations.pytorch import ToTensorV2

def augment(image_size):
    return A.Compose([A.Resize(image_size, image_size)])