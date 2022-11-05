from utils.utils import create_df, create_log_dir, seed_everything, submission
from train.train import train
from utils.ensemble import predict
from utils.config import Config

if __name__ == '__main__':

    seed_everything(Config.seed)
    create_log_dir(Config.log_dir) 
    create_log_dir(Config.sub_file)
    create_df(Config)

    for fold in range(Config.folds):
        train(args=Config, fold_num=fold)
    
    argmax= predict(Config)
    submission(Config, argmax)



