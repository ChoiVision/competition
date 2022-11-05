import timm

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from utils.custom_scheduler import CustomCosineAnnealingWarmUpRestarts
from utils.arcface import ArcMarginProduct
from utils.loss import FocalLoss

import timm

class Network(pl.LightningModule):
  def __init__(self, args):
    super().__init__()
    self.args= args
    self.model= timm.create_model(self.args.model_name, pretrained= self.args.pretrained, drop_rate= self.args.drop_rate)
    self.embedding = nn.Linear(self.model.get_classifier().in_features, self.args.embedding_size)
    self.model.reset_classifier(num_classes=0, global_pool="avg")

    self.arc = ArcMarginProduct(
        in_features= self.args.embedding_size,
        out_features= self.args.num_classes,
        s= self.args.arc_s,
        m= self.args.arc_m,
        easy_margin= self.args.arc_easy_margin,
        ls_eps= self.args.arc_ls_eps,
    )

    self.loss_fn= FocalLoss(args.num_classes)

  def forward(self, image):
    features= self.model(image)
    embedding= self.embedding(features)

    return embedding

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr = self.args.learning_rate, weight_decay=self.args.weight_decay)
    scheduler = CustomCosineAnnealingWarmUpRestarts(optimizer, T_0=self.args.t0, T_mult =self.args.tmult, eta_max = self.args.eta_max)

    return [optimizer], [scheduler]

  def training_step(self, batch, batch_idx):
    return self._step(batch, 'train')

  def validation_step(self, batch, batch_idx):
    return self._step(batch,' val')

  def validation_epoch_end(self, outputs):

    avg_loss = torch.stack([x for x in outputs]).mean()
    log = {'val_loss': avg_loss}

  def _step(self, batch, step):
    image, label= batch['image'], batch['target']
    embedding= self(image)
    output= self.arc(embedding, label)
    loss= self.loss_fn(output, label)

    self.log(f'{step}_loss', loss)

    return loss