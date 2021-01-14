import os
import pandas as pd
import numpy as np

import config
import transforms
import metrics
import dataset
import engine
from model import UneXt50
from scheduler import GradualWarmupSchedulerV2
from radam import RAdam
from lookahead import Lookahead

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import cv2

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from apex import amp

from torchcontrib.optim import SWA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler


def run():
    d = {'image_id': os.listdir(config.TRAIN_IMAGE_PATH), 'mask_id': os.listdir(config.TRAIN_MASK_PATH)}
    df = pd.DataFrame(data=d)

    folds = df.copy()

    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(folds)):

        print(f'FOLD: {fold+1}/{config.N_FOLDS}')

        train_test = folds.iloc[train_idx]
        valid_test = folds.iloc[valid_idx]

        train_test.reset_index(drop=True, inplace=True)
        valid_test.reset_index(drop=True, inplace=True)

        train_dataset = dataset.HuBMAPDataset(train_test, transforms=transforms.transforms_train)
        train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

        valid_dataset = dataset.HuBMAPDataset(valid_test, transforms=transforms.transforms_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

        loss_history = {
            "train": [],
            "valid": []
        }

        dice_history = {
            "train": [],
            "valid": []
        }

        jaccard_history = {
            "train": [],
            "valid": []
        }

        dice_max = 0.0
        kernel_type = 'unext50'
        best_file = f'../drive/MyDrive/{kernel_type}_best_fold{fold}_strong_aug_70_epochs.bin'

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = UneXt50().to(device)
        optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, model.parameters()),lr=config.LR), alpha=0.5, k=5)
        # base_opt = optim.Adam(model.parameters(), lr=3e-4)
        # optimizer = SWA(base_opt)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.N_EPOCHS)
        # scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=config.WARMUP_EPO, after_scheduler=scheduler_cosine)
        loss_fn = metrics.symmetric_lovasz

        for epoch in range(config.N_EPOCHS):
            
            scheduler.step(epoch)
            avg_train_loss, train_dice_scores, train_jaccard_scores = engine.train_loop_fn(model, 
                                                                                           train_loader, 
                                                                                           optimizer, 
                                                                                           loss_fn,
                                                                                           metrics.dice_coef_metric, 
                                                                                           metrics.jaccard_coef_metric,
                                                                                           device)
                    
            # if epoch > 10 and epoch % 5 == 0:
            #   optimizer.update_swa()
            
            loss_history["train"].append(avg_train_loss)
            dice_history["train"].append(train_dice_scores)
            jaccard_history["train"].append(train_jaccard_scores)


            avg_val_loss, val_dice_scores, val_jaccard_scores = engine.val_loop_fn(model, 
                                                                                   valid_loader, 
                                                                                   optimizer, 
                                                                                   loss_fn,
                                                                                   metrics.dice_coef_metric, 
                                                                                   metrics.jaccard_coef_metric,
                                                                                   device)


            loss_history["valid"].append(avg_val_loss)
            dice_history["valid"].append(val_dice_scores)
            jaccard_history["valid"].append(val_jaccard_scores)

            print(f"Epoch: {epoch+1} | lr: {optimizer.param_groups[0]['lr']:.7f} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")
            print(f"train dice: {train_dice_scores:.4f} | val dice: {val_dice_scores:.4f} | train jaccard: {train_jaccard_scores:.4f} | val jaccard: {val_jaccard_scores:.4f}")

            if val_dice_scores > dice_max:
                print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(dice_max, val_dice_scores))
                torch.save(model.state_dict(), best_file)
                dice_max = val_dice_scores

        # optimizer.swap_swa_sgd()

if __name__ == "__main__":
    run()