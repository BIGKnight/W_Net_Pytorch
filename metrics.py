import torch
import torch.nn as nn

class JointLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(JointLoss, self).__init__()
        self.MSELoss = nn.MSELoss(size_average=True)
        self.BCELoss = nn.BCELoss(size_average=True)
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x, gt_map, target_map):
        return self.MSELoss(x, gt_map) * alpha + self.BCELoss(x, target_map) * beta
    
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
