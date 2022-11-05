import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
  def __init__(
      self,
      in_features: int,
      out_features: int,
      s: float,
      m: float,
      easy_margin: bool,
      ls_eps: float,
  ):

    super(ArcMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.ls_eps = ls_eps
    self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight)

    self.easy_margin = easy_margin
    self.cos_m = math.cos(m)
    self.sin_m = math.sin(m)
    self.th = math.cos(math.pi - m)
    self.mm = math.sin(math.pi - m) * m

  def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    cosine = cosine.to(torch.float32)

    sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
    phi = cosine * self.cos_m - sine * self.sin_m
    
    if self.easy_margin:
      phi = torch.where(cosine > 0, phi, cosine)
    
    else:
      phi = torch.where(cosine > self.th, phi, cosine - self.mm)
    
    one_hot = torch.zeros(cosine.size(), device=device)
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    
    if self.ls_eps > 0:
      one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
    
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    output *= self.s

    return output