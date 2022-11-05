import argparse
from utils.utils import preprocessing_df, create_log_dir
from train.train import train
from train.inference import infer

if __name__ == '__main__':
    def argparse():
        parser = argparse.ArgumentParser(description='Kaggle Happy Whale')

        parser.add_argument('--seed', type= int, default= 42)
        parser.add_argument('--image_size', type= int, default= 820)
        parser.add_argument('--folds', type= int, default= 10)
        parser.add_argument('--default_dir', type= str, default= './')
        parser.add_argument('--train_dir', type= str, default= './train_images/')
        parser.add_argument('--test_dir', type= str, default= './test_images/')
        parser.add_argument('--submission', type= str, default= './sample_submission.csv')
        parser.add_argument('--log_dir', type= str, default= './log_dir/')
        parser.add_argument('--model_name', type= str, default= 'tf_efficientnet_b6')
        parser.add_argument('--embedding_size', type=int, default= 512)
        parser.add_argument('--drop_rate', type=float, default= 0.3)
        parser.add_argument('--num_workers', type= int, default= 4)
        parser.add_argument('--num_classes', type= int, default= 15587)
        parser.add_argument('--val_fold', type= int, default= 3)
        parser.add_argument('--batch_size', type = int, default=4)
        parser.add_argument('--learning_rate', type= float, default= 5e-7)
        parser.add_argument('--k', type=int, default= 50)
        parser.add_argument('--t0', type= int, defualt= 5)
        parser.add_argument('--tmult', type= int, defualt= 3)
        parser.add_argument('--eta_max', type= float, defualt= 1e-2)
        parser.add_argument('--arc_s', type=float, default= 30.0)
        parser.add_argument('--arc_m', type=float, default= 0.5)
        parser.add_argument('--arc_easy_margin', type=bool, default= False)
        parser.add_argument('--arc_ls_eps', type=int, default= 0)
        parser.add_argument('--accelerator', type= str, default='gpu')
        parser.add_argument('--device', type= int, default= -1)
        parser.add_argument('--half', type= int, default= 16)
        parser.add_argument('--pretrained', type= bool, default= True)
        parser.add_argument('--accumulate', type= int, default= 16)

        args = parser.parse_args()

        return args

    args = argparse()

    train = preprocessing_df(args, 'train')
    test = preprocessing_df(args, 'test')

    create_log_dir(args.log_dir)

    train(args=args)
    infer(args=args)
