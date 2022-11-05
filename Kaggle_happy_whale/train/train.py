import pytorch_lightning as pl

from model.model import Network
from datamodules.datamodule import CustomDataModule

def train(args):
  dm= CustomDataModule(args)
  dm.set_fold_num(args.val_fold)

  model= Network(args)

  model_checkpoint= pl.callbacks.ModelCheckpoint(
      monitor= ' val_loss',
      dirpath= f'{args.log_dir}_{args.val_fold}/',
      mode= 'min'
      )
  
  early_stop= pl.callbacks.EarlyStopping(
      monitor= ' val_loss',
      patience= args.patience,
      verbose= True,
      mode= 'min'
    )

  train_dict = dict(
      accelerator= args.accelerator, 
      devices= args.device,
      callbacks = [model_checkpoint, early_stop],
      precision = args.half,
      max_epochs = args.epochs
      )
  
  trainer= pl.Trainer(**train_dict, check_val_every_n_epoch=1)
  trainer.tune(model, dm)
  trainer.fit(model, dm)