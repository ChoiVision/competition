import timm
import numpy as np
import torch
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl

from utils.custom_scheduler import CustomCosineAnnealingWarmUpRestarts
from utils.cutmix import cutmix
from utils.loss import FocalLoss


class Network(pl.LightningModule):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.model = timm.create_model(model_name = args.model, pretrained = args.pretrained, num_classes = args.num_classes)
    self.f1 = torchmetrics.F1Score(num_classes = args.num_classes, threshold = 0.5, average = 'macro')
    self.loss = FocalLoss()
    self.values = list()
    
  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr = self.args.learning_rate, weight_decay=self.args.weight_decay)
    scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer, T_0=self.args.t0, T_mult =self.args.tmult, eta_max = self.args.eta_max)
    return [optimizer], [scheduler]

  def training_step(self, batch, batch_idx, args):
    image, label= batch

    if np.random.rand() < 0.5:
      image, image_labels = cutmix(image, label, 1.)
      logits= self(image)
      loss = self.loss(logits, image_labels[0]) * image_labels[2] + self.loss(logits, image_labels[1]) * (1. - image_labels[2])
      f1= self.f1(logits, image_labels)
    
    else:
      logits= self(image)
      loss= self.loss(logits)
      f1= self.f1(logits, label)
    
    self.log_dict(
            {
                'train_loss':loss,
                'train_f1':f1
            },
            prog_bar = True
                    )


    return loss

  def validation_step(self, batch, batch_idx):
    return self._step(batch, step='valid')

  def validation_epoch_end(self, outputs): 
    avg_loss= torch.stack([x for x in outputs]).mean()
    self.log('avg_valid_loss', avg_loss)

  def test_step(self, batch, batch_idx):
    image = batch
    logits = self(image)
    value = torch.softmax(logits,1).detach().cpu().numpy()
    self.values.extend(value)    

  def _step(self, batch, step):
    image, label= batch
    output= self(image)
    loss= self.loss(label, output)
    f1 = self.f1(output, label)
    self.log_dict(
        {
            f'{step}_loss':loss,
            f'{step}_f1':f1
        },
        prog_bar = True,
        logger = True
    )

    return loss
