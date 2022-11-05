class Config:
    train_csv= 'data/train.csv'
    test_csv= 'data/test.csv'
    log_dir= 'log'
    sub_file= 'data/submission.csv'
    submission= 'submission/'

    folds= 5
    image_size= 420
    num_classes= 7
    batch_size= 32
    accumulate= 2
    half= 16
    epochs= 50
    model= 'tf_efficientnet_b0'
    pretrained= True

    smoothing = .2
    weight_decay= 5e-6
    t0= 10
    tmult= 3
    eta_max= 1e-3
    tup= 4
    gamma= .4

    accelerator= 'gpu'
    device= -1
    num_workers= 4

    seed= 42
    patience= 18