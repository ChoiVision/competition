import torch
import torch.nn as nn
import torch.nn.functional as F
from dl.model.common import ConvBn, ConvBnAct

class LinearBottleNeck(nn.Module):
    def __init__(self, in_c, out_c, s, num_classes, t= 6):
        super().__init__()
        self.in_c= in_c
        self.out_c= out_c
        self.s= s
        
        self.residual= nn.Sequential(
            ConvBnAct(in_c, in_c * t, 1),
            ConvBnAct(in_c * t, in_c * t, 3, s, g= in_c * t),
            ConvBn(in_c * t, out_c, 1)
        )
        
    def forward(self, x):
        residual= self.residual(x)
        
        if self.s == 1 and self.in_c == self.out_c:
            residual += x
            
        return residual
