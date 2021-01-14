import cv2 
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

transforms_train = A.Compose([
  A.Resize(256, 256, p=1.0),
  A.HorizontalFlip(p=0.5),
  A.VerticalFlip(p=0.5),
  A.Transpose(p=0.5),
  A.RandomRotate90(p=0.5),
  A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),

  A.IAAAdditiveGaussianNoise(p=0.2),
  A.IAAPerspective(p=0.5),

  A.OneOf(
      [
          A.CLAHE(p=1),
          A.RandomBrightness(p=1),
          A.RandomGamma(p=1),
      ],
      p=0.9,
  ),

  A.OneOf(
      [
          A.IAASharpen(p=1),
          A.Blur(blur_limit=3, p=1),
          A.MotionBlur(blur_limit=3, p=1),
      ],
      p=0.9,
  ),

  A.OneOf(
      [
          A.RandomContrast(p=1),
          A.HueSaturationValue(p=1),
      ],
      p=0.9,
  ),
  
  A.Compose([
      A.VerticalFlip(p=0.5),              
      A.RandomRotate90(p=0.5)]
  )

])


transforms_valid = A.Compose([
    A.Resize(256, 256, p=1.0),

])