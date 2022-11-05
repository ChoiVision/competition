import pytorch_lightning as pl
from datamodules.datamodule import CustomDataModule 
from model.model import Network

def train(args, fold_num):
  dataModule = CustomDataModule(args)
  dataModule.set_fold_num(fold_num)
  dataModule.setup('fit')
  
  model = Network(args)

  model_checkpoint = pl.callbacks.ModelCheckpoint(
      monitor = 'valid_loss', 
      dirpath = f'{args.log_dir}/{fold_num}', 
      mode = 'min')
  
  earlystop = pl.callbacks.EarlyStopping(
      monitor='valid_loss',
      patience = args.patience,
      verbose=True,
      mode='min') 
  

  train_dict = dict(
      accelerator= args.accelerator, 
      devices= args.device,
      callbacks = [model_checkpoint, earlystop],
      precision = args.precision,
      max_epochs = args.epochs
      )

  trainer = pl.Trainer(**train_dict)
  trainer.fit(model, dataModule)