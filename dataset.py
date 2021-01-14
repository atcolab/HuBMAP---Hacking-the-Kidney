import cv2
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset

import config

class HuBMAPDataset(Dataset):
  def __init__(self, df, transforms=None):
    self.df = df
    self.transforms = transforms
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    image_id = self.df['image_id'].values[index]
    image_path = f'{config.TRAIN_IMAGE_PATH}/{image_id}'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_id = self.df['mask_id'].values[index]
    mask_path = f'{config.TRAIN_MASK_PATH}/{mask_id}'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if self.transforms:
      augmented = self.transforms(image=image, mask=mask)
      image = augmented['image']
      mask = augmented['mask']

    image = image.astype(np.float32)
    image /= 255.0
    image = image.transpose(2, 0, 1)

    return torch.tensor(image), torch.tensor(mask)