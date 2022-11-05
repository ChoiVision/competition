import pytorch_lightning as pl
from datamodules.datamodule import CustomDataModule 
from model.model import Network

def test(ckpt ,args, fold_num):
  dataModule = CustomDataModule(args)
  dataModule.set_fold_num(fold_num)
  dataModule.setup('test')

  model = Network(args).load_from_checkpoint(checkpoint_path = ckpt, args = args)

  train_dict = dict(
      accelerator= args.accelerator, 
      devices= args.device,
      precision = args.half,
      accumulate_grad_batches = args.accumulate,
      auto_lr_find = True, 
      max_epochs = args.epochs
  )

  trainer = pl.Trainer(**train_dict) 

  trainer.test(model, dataModule)
  
  return model.values