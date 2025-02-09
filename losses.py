#import math 
import torch 
import torch.nn as nn 
# import torch.nn.functional as F
# from torch.nn import Parameter


class ASDLoss(nn.Module):
    def __init__(self, reduction=True):
        super(ASDLoss, self).__init__()
        if reduction == True:
            self.ce = nn.CrossEntropyLoss()
        
        else:
            self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss
    
#可以替换ASDLoss
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.ce = nn.BCELoss()

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss
    
#可以替换ASDLoss
class BCELogitLoss(nn.Module):
    def __init__(self):
        super(BCELogitLoss, self).__init__()
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss