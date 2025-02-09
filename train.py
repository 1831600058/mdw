import random 
import torch 
import numpy as np
from dataloader import train_dataset, test_dataset
from torch.utils.data import DataLoader
from utils import dataset_split
from trainer import Trainer
import yaml
import argparse


def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    #name_list = ['pump', 'valve']
    #data_path = '/ddnstor/imu_tlsz1/lsz/project/origin/STgram-MFN-main/data/dataset'
    #eval_path = '/ddnstor/imu_tlsz1/lsz/project/origin/STgram-MFN-main/data/eval_dataset'
    root_path ='/ddnstor/imu_tlsz1/lsz/project/DualAttentionMFN/data/dataset'
    device_num = cfg['gpu_num']
    #device = torch.device(f'cuda:{device_num}')
    if device_num is None:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_num}')
    
    print('training dataset loading...')
    dataset = train_dataset(root_path, name_list)
    
    train_ds, valid_ds = dataset_split(dataset, split_ratio=cfg['split_ratio'])
    
    train_dataloader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg['batch_size'])
    
    trainer = Trainer(device=device, alpha=cfg['alpha'], mode=cfg['mode'],
                      epochs=cfg['epoch'], class_num=cfg['num_classes'],
                      m=cfg['m'], lr=cfg['lr'])
    
    trainer.train(train_dataloader, valid_dataloader, cfg['save_path'])

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    random_seed(seed=2025)
    torch.set_num_threads(8)
    main()