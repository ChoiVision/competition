import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p

class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, k, s= None, p= None, g= 1):
        super().__init__()
        self.conv= nn.Conv2d(in_c, out_c, k, s, autopad(k, p), groups= g, bias= False)
        self.bn= nn.BatchNorm1d(out_c)
    
    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        
        return x
    
       
class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k, s= None, p= None, g= 1, act= True):
        super().__init__()
        self.conv= nn.Conv2d(in_c, out_c, k, s, autopad(k, p), groups= g, bias= False)
        self.bn= nn.BatchNorm1d(out_c)
        self.act= nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity)
    
    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        x= self.act(x)
        
        return x