import argparse
from train.train import train
from utils.ensemble import ensemble_predict
from utils.utils import *


if '__name__' == '__main__':
    def parser():
        parser = argparse.ArgumentParser(description='LG DACON CONTEST')
        parser.add_argument('--train_csv', type=str, default='./data/train/*/*.csv', help = 'glob train csv file')
        parser.add_argument('--train_df', type=str, default='./data/train.csv', help = 'train data frame')
        parser.add_argument('--test_df', type=str, default='./data/test.csv', help = 'test data frame')
        parser.add_argument('--train_image', type=str, default='./data/train/*/*.jpg', help = 'glob train csv file')
        parser.add_argument('--train_json', type=str, default='./data/train/*/*.json', help = 'glob train csv file')
        parser.add_argument('--test_image', type=str, default='./data/test/*/*.jpg', help = 'glob train csv file')
        parser.add_argument('--sub_file', type=str, default='./data/sample_subission.csv')
        parser.add_argument('--submission', type=str, default='./submssion/')
        parser.add_argument('--default_dir', type=str, default='./', help='default dir')
        parser.add_argument('--model_name', type=str, default='resnext50_32x4d', help='timm model list')
        parser.add_argument('--image_size', type=int, default=520, help='Image Size')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
        parser.add_argument('--epochs', type=int, default=100, help='Epoch')  
        parser.add_argument('--num_classes', type=int, default=25, help='Classes') 
        parser.add_argument('--half', type=int, default=16, help='precision') 
        parser.add_argument('--learning_rate', type=float, default=4e-5, help='Learning Rate')
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument('--t0', type=float, default=5)
        parser.add_argument('--tup', type=float, default=5)
        parser.add_argument('--tmult', type=float, default=2)
        parser.add_argument('--eta_max', type=float, default=0.01)
        parser.add_argument('--weight_decay', type=float, default=3e-5)
        parser.add_argument('--patience', type=int, default=12, help='Early Stopping patience')
        parser.add_argument('--folds', type=int, default=7, help='Stratified Kfold')
        parser.add_argument('--seed', type=int, default=42, help='Fix Seed')
        parser.add_argument('--log_dir', type=str, default='./log/resnext')
        parser.add_argument('--accelerator', type=str , default='gpu', help = 'set gpu or cpu')
        parser.add_argument('--device', type=int , default=-1)
        parser.add_argument('--precision', type=int , default=16)
        parser.add_argument('--pretrained', type=bool , default=True)
        parser.add_argument('--accumulate', type=int , default=8, help = 'batch accumulate')

        args = parser.parse_args()

        return args

    args = parser()

    seed_everything(args.seed)
    create_log_dir(args.log_dir)
    create_log_dir(args.submssion)
    label_index, train_df, test_df = create_dataframe(args)

    for fold_num in range(args.folds):
        train(args=args, fold_num=fold_num)

    label = ensemble_predict(label_index, args)

    submission(args, label)
