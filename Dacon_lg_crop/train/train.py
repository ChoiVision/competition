import pytorch_lightning as pl
from datamodules.datamodule import CustomDataModule
from model.model import Network

def train(args, fold_num):
  dataModule = CustomDataModule(args)
  dataModule.set_fold_num(fold_num)
  dataModule.setup('fit')
  
  model = Network(args)
  
  earlystop = pl.callbacks.EarlyStopping(
      monitor='valid_loss',
      patience = args.patience,
      verbose=True,
      mode='min') 
  
  train_dict = dict(
      accelerator= args.accelerator, 
      devices= args.device,
      callbacks = [earlystop],
      precision = args.half,
      max_epochs = args.epochs
      )

  trainer = pl.Trainer(**train_dict)

  trainer.fit(model, dataModule)

  trainer.save_checkpoint(f'{args.log_dir}/fold:{fold_num}_best.ckpt')