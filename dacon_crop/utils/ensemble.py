import glob
import numpy as np
from train.eval import test

def predict(args):
  ensemble = list()  
  ckpt_list = sorted(glob.glob(args.log_dir + '/*/*.ckpt'))
  fold_list = list(range(args.folds))
  
  for fold, ckpt in zip(fold_list, ckpt_list):
    result = test(ckpt, args, fold)
    result = np.array(result)
    ensemble.append(result)

  final = np.mean(ensemble, axis =0)
  argmax = np.argmax(final, 1)
  
  return argmax