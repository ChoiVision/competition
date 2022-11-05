import glob
import numpy as np
from train.inference import test

def ensemble_predict(labels, args):
    ensemble = list()
    ckpt_list = sorted(glob.glob(f'{args.log_dir}/*.ckpt'))
    fold_list = list(range(args.folds))

    for fold, ckpt in zip(fold_list, ckpt_list):
        result = test(ckpt, args, fold)
        result = np.array(result)
        ensemble.append(result)

    final = np.mean(ensemble, axis =0)
    argmax = np.argmax(final, 1)
    label = [labels[i] for i in argmax]

    return label