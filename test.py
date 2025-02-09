import torch 
from torch.utils.data import DataLoader
from sklearn import metrics
from net import MixData_WeightProcess
from losses import ASDLoss
from losses import BCELoss
from losses import BCELogitLoss
from dataloader import test_dataset  
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np


def evaluator(net, test_loader, criterion, device):
    net.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_wavs, x_mels, labels, AN_N_labels in test_loader:
            x_wavs, x_mels, labels, AN_N_labels = x_wavs.to(device), x_mels.to(device), labels.to(device), AN_N_labels.to(device)
            
            logits, _ = net(x_wavs, x_mels, labels, train=False)
            
            score = criterion(logits, labels)

            # y_pred.extend(score.tolist())
            # y_true.extend(AN_N_labels.tolist())
            y_pred.extend([score.item()])
            y_true.extend([AN_N_labels.item()])
    
    auc = metrics.roc_auc_score(y_true, y_pred)
    pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    return auc, pauc                
        
def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    print(cfg)
    
    device_num = cfg['gpu_num']
    
    device = torch.device(f'cuda:{device_num}')

    net = MixData_WeightProcess(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    
    net.load_state_dict(torch.load(cfg['save_path']))
    net.eval()
    
    # criterion = ASDLoss(reduction=False).to(device)
    criterion = ASDLoss().to(device)
    #criterion = BCELoss().to(device)
    
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    #name_list = ['pump', 'valve']
    #data_path = '/ddnstor/imu_tlsz1/lsz/project/origin/STgram-MFN-main/data/dataset'
    #eval_path = '/ddnstor/imu_tlsz1/lsz/project/origin/STgram-MFN-main/data/eval_dataset'
    root_path ='/ddnstor/imu_tlsz1/lsz/project/DualAttentionMFN/data/dataset'
    
    avg_AUC = 0.
    avg_pAUC = 0.
    
    for i in range(len(name_list)):
        test_ds = test_dataset(root_path, name_list[i], name_list)
        test_dataloader = DataLoader(test_ds, batch_size=1)
        
        AUC, PAUC = evaluator(net, test_dataloader, criterion, device)
        avg_AUC += AUC 
        avg_pAUC += PAUC 
        
        print(f"{name_list[i]} - AUC: {AUC:.4f}, pAUC: {PAUC:.4f}")
    
    avg_AUC = avg_AUC / len(name_list)
    avg_pAUC = avg_pAUC / len(name_list)
    
    print(f"Average AUC: {avg_AUC:.4f},  Average pAUC: {avg_pAUC:.4f}")
        
    
if __name__ == '__main__':
    torch.set_num_threads(8)
    main()