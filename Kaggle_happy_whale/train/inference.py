import pandas as pd
import numpy as np
from utils.inference_utils import *

def infer(args):
  module= load_eval_module(args=args, checkpoint_path='./log_9/epoch=0-step=11483.ckpt')
  train_dl, val_dl,test_dl= load_dataloaders(args)
  encoder = load_encoder(args)
  
  train_image_names, train_embeddings, train_targets = get_embeddings(module, train_dl, encoder, stage="train")
  val_image_names, val_embeddings, val_targets = get_embeddings(module, val_dl, encoder, stage="val")
  test_image_names, test_embeddings, test_targets = get_embeddings(module, test_dl, encoder, stage="test")

  D, I = create_and_search_index(args.embedding_size, args.train_embeddings, args.val_embeddings, args.k)  # noqa: E741

  val_targets_df = create_val_targets_df(train_targets, val_image_names, val_targets)

  val_df = create_distances_df(val_image_names, train_targets, D, I, "val")

  best_th, best_cv = get_best_threshold(val_targets_df, val_df)

  train_embeddings = np.concatenate([train_embeddings, val_embeddings])
  train_targets = np.concatenate([train_targets, val_targets])

  D, I = create_and_search_index(args.embedding_size, train_embeddings, test_embeddings, args.k) 

  test_df = create_distances_df(test_image_names, train_targets, D, I, "test")

  predictions = create_predictions_df(test_df, best_th)

  public_predictions = pd.read_csv(args.submission)
  ids_without_backfin = np.load(args.default_dir + 'encoder_classes.npy', allow_pickle=True)

  ids2 = public_predictions["image"][~public_predictions["image"].isin(predictions["image"])]

  predictions = pd.concat(
      [
          predictions[~(predictions["image"].isin(ids_without_backfin))],
          public_predictions[public_predictions["image"].isin(ids_without_backfin)],
          public_predictions[public_predictions["image"].isin(ids2)],
      ]
  )
  predictions = predictions.drop_duplicates()

  predictions.to_csv(args.submission, index=False)