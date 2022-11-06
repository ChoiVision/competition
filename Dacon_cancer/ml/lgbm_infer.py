import pandas as pd
import numpy as np

import lightgbm
from lightgbm import log_evaluation, early_stopping
from config import Config
from ml_utils.ml_util import preprocess_csv, create_skf

params= {'n_estimators': 4008, 'max_depth': 352, 'lambda_l1': 3.053345198992793, 'lambda_l2': 5.699878131556061, 'num_leaves': 383, 'max_leaf_nodes': 98, 'feature_fraction': 0.9998269180212589, 'bagging_fraction': 0.7911027185332159, 'bagging_freq': 93, 'min_child_samples': 29, 'learning_rate': 0.007424496866670218}
test= preprocess_csv(Config.TEST_CSV, 'test')

train= preprocess_csv(Config.TRAIN_CSV, 'train')
test= preprocess_csv(Config.TEST_CSV, 'test')
train= create_skf(config= Config, df= train)

target= train['N_category']
features= train.drop(['N_category','kfold'], axis=1)

pred= list()
pred_proba= list()
for k in range(Config.FOLDS):

    print("#################################")
    print(f'######### {k + 1} FOLD START ##########')
    print("#################################")

    train_idx= train['kfold'] != k
    valid_idx= train['kfold'] == k

    X_train= features[train_idx].values
    y_train= target[train_idx].values

    X_valid= features[valid_idx].values
    y_valid= target[valid_idx].values


    model= lightgbm.LGBMClassifier(**params)
    callbacks= [log_evaluation(period=500), early_stopping(stopping_rounds=Config.STOP)]

    model.fit(X_train, y_train, 
    eval_set= [(X_valid, y_valid)], 
    eval_metric='binary_logloss', 
    callbacks= callbacks)


    pred_proba.append(model.predict_proba(test))

np_= np.array(pred_proba)
sum_= np.sum(np_, axis=0)
np.save('ml_logs/pred_proba.npy', sum_)
res= np.argmax(sum_, 1)
sub= pd.read_csv(Config.SUB_CSV)
sub['N_category']= res
sub.to_csv('lgbm.csv', index= False)