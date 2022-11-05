import cv2
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, df, augmentation):
        self.df = df
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.img_path[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = self.df.disease_code[index]

        if self.augmentation is not None:
            image = self.augmentation(image = image)['image']

        return image, label

class TestDataset(Dataset):
    def __init__(self, df, augmentation):
        self.df = df
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.image_path[index]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if self.augmentation is not None:
            image = self.augmentation(image = image)['image']

        return image