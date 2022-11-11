import pandas as pd
from sklearn.model_selection import StratifiedKFold

def preprocess_csv(path, mode='train'):
    df= pd.read_csv(path)
    if mode == 'train':
        # drop_cols= ['ID', 'img_path', 'mask_path', '수술연월일']
        drop_cols= ['수술연월일']
        df.drop(drop_cols, axis=1, inplace= True)
    
        return df

    if mode == 'test':
        # drop_cols= ['ID', 'img_path', '수술연월일']
        drop_cols= ['수술연월일']
        df.drop(drop_cols, axis=1, inplace= True)

        return df

def create_skf(config, df):
    skf= StratifiedKFold(n_splits=config.FOLDS, shuffle= True, random_state= config.SEED)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X= df, y= df['N_category'])):
        df.loc[valid_idx, 'kfold']= int(fold)


    return df
    
def concat_df(train, test):
    apps= pd.concat([train, test])

    return apps

def split_train_test(apps):
    train= apps[~apps['N_category'].isnull()]

    test= apps[apps['N_category'].isnull()]
    test.drop('N_category', axis=1)

    return train, test
