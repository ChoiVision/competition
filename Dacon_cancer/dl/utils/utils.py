import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl

def seed_everything(seed):
    pl.seed_everything(seed)
    
def create_file(file):
    if os.path.exists(file):
        print('File Exist')
    else:
        print(f'Create {file}')
        os.mkdir(file)
        
def file_preprocessing():
    pass

