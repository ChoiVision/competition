import lightgbm
from lightgbm import log_evaluation, early_stopping

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

from config import Config
from ml_utils.ml_util import preprocess_csv, create_skf

train= preprocess_csv(Config.TRAIN_CSV, 'train')
test= preprocess_csv(Config.TEST_CSV, 'test')
train= create_skf(config= Config, df= train)

target= train['N_category']
features= train.drop(['N_category','kfold'], axis=1)


def objective(trial: Trial):

    params= {
    'objective': 'binary',
    'boosting_type' : 'gbdt',
    "n_estimators" : trial.suggest_int('n_estimators', 100, 2500),
    'max_depth':trial.suggest_int('max_depth', 4, 128),
    'seed': Config.SEED,
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
    'num_leaves': trial.suggest_int('num_leaves', 4, 64),
    'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 4, 512),
    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 1.0),
    'bagging_freq': trial.suggest_int('bagging_freq', 1, 12),
    'min_child_samples': trial.suggest_int('min_child_samples', 4, 128),
    'learning_rate': trial.suggest_float('learning_rate', 6e-9, 1e-2),
    'n_jobs': -1,
    }
    
    callbacks= [log_evaluation(period=500), early_stopping(stopping_rounds=Config.STOP)]

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
        model.fit(X_train, y_train, 
        eval_set= [(X_valid, y_valid)], 
        eval_metric='binary_logloss', 
        callbacks= callbacks)

        log= model.predict_proba(test, num_iteration= model.best_iteration_)
        pred= model.predict(test)

#         pred= model.predict(X_valid, num_iteration= model.best_iteration_)
#         print()
#         print("PRED")
#         print(pred)
#         print()

sampler= TPESampler(seed=Config.SEED)
study= optuna.create_study(direction='minimize', sampler= sampler)
study.optimize(objective, n_trials=3)
print('##############################################')
print('Number of finished trials:', len(study.trials))
print("Best Score: ", study.best_value)
print('Best trial: ', study.best_trial.params)
print('##############################################')




# print(objective(trail))