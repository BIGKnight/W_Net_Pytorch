import torch
import torch.nn as nn
import sys

class JointLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(JointLoss, self).__init__()
        self.MSELoss = nn.MSELoss(size_average=False)
        self.BCELoss = nn.BCELoss(size_average=False)
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x, gt_map, target_map):
        mse = self.MSELoss(x, gt_map) * self.alpha
        bce = self.BCELoss(x, target_map) * self.beta
        sys.stdout.write("mse loss = {}, bce loss = {}\r".format(mse, bce))
        sys.stdout.flush()
        return  mse + bce
    
class AEBatch(nn.Module):
    def __init__(self):
        super(AEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.abs(torch.sum(estimated_density_map - gt_map, dim=(1, 2, 3)))


class SEBatch(nn.Module):
    def __init__(self):
        super(SEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.pow(torch.sum(estimated_density_map - gt_map, dim=(1, 2, 3)), 2)
