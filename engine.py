import torch
from tqdm import tqdm
from apex import amp
import numpy as np

def train_loop_fn(model, loader, optimizer, loss_func, dice_coef_metric, jaccard_coef_metric, device):

  model.train()

  losses = []
  dice = []
  jaccard = []

  for step, (data, target) in tqdm(enumerate(loader), total=len(loader)):
    data = data.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)

    optimizer.zero_grad()

    outputs = model(data)
    probs = torch.sigmoid(outputs)

    loss = loss_func(outputs, target.unsqueeze(1))

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    dice_scores = dice_coef_metric(probs.squeeze(1).detach().cpu(), target.detach().cpu(), 0.5)
    jaccard_scores = jaccard_coef_metric(probs.squeeze(1).detach().cpu(), target.detach().cpu(), 0.5)

    if ((step + 1) % 1 == 0) or ((step + 1) == len(loader)):
      optimizer.step()

    losses.append(loss.item())
    dice.append(dice_scores)
    jaccard.append(jaccard_scores)

  return np.array(losses).mean(), np.array(dice).mean(), np.array(jaccard).mean()


def val_loop_fn(model, loader, optimizer, loss_func, dice_coef_metric, jaccard_coef_metric, device):

  model.eval()

  losses = []
  dice = []
  jaccard = []

  with torch.no_grad():
    for (data, target) in tqdm(loader):
      data = data.to(device, dtype=torch.float)
      target = target.to(device, dtype=torch.float)

      outputs = model(data)
      probs = torch.sigmoid(outputs)

      loss = loss_func(outputs, target.unsqueeze(1))

      dice_scores = dice_coef_metric(probs.squeeze(1).detach().cpu(), target.detach().cpu(), 0.5)
      jaccard_scores = jaccard_coef_metric(probs.squeeze(1).detach().cpu(), target.detach().cpu(), 0.5)

      losses.append(loss.item())
      dice.append(dice_scores)
      jaccard.append(jaccard_scores)

  return np.array(losses).mean(), np.array(dice).mean(), np.array(jaccard).mean()