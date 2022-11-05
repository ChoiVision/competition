import numpy as np
import timm

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
import torchmetrics

from utils.label_smoothing import LabelSmoothingLoss
from utils.custom_scheduler import CosineAnnealingWarmUpRestarts

class Network(pl.LightningModule):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.model = timm.create_model(model_name = args.model, pretrained = args.pretrained, num_classes = args.num_classes)
    self.f1 = torchmetrics.F1Score(num_classes = args.num_classes, threshold = 0.5, average = 'macro')
    self.loss = LabelSmoothingLoss(classes= args.num_classes, smoothing= args.smoothing, dim=-1)
    self.values = list()

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr = self.args.learning_rate, weight_decay=self.args.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=self.args.t0, T_mult =self.args.tmult, eta_max = self.args.eta_max, T_up = self.args.tup, gamma=self.args.gamma)
    return [optimizer], [scheduler]

  def training_step(self, batch, batch_idx):
    return self._step(batch, step='train')

  def validation_step(self, batch, batch_idx):
    return self._step(batch, step='valid')

  def _step(self, batch, step):
    image, label= batch
    logits= self(image)
    loss= self.loss(logits, label)
    f1= self.f1(logits, label)
    self.log_dict(
      {
        f'{step}_loss':loss,
        f'{step}_f1':f1
      },
      prog_bar= True
    )

    return loss

  def validation_epoch_end(self, outputs):
    avg_loss= torch.stack([x for x in outputs]).mean()
    self.log('avg_valid_loss', avg_loss)

  def test_step(self, batch, batch_idx):
    image = batch
    logits = self(image)
    value = torch.softmax(logits,1).detach().cpu().numpy()
    self.values.extend(value)    