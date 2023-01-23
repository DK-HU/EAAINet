from __future__ import division
import torch
import torch.nn as nn
import numpy as np

class EuclideanLoss_with_Uncertainty(nn.Module):
    def __init__(self):
        super(EuclideanLoss_with_Uncertainty, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, pred, target, mask, certainty):
        loss_reg = self.pdist(pred, target)
        certainty_map = torch.max(certainty.cuda(), torch.tensor(1e-6).cuda())

        loss_map = 3 * torch.log(certainty_map) + loss_reg / (2 * certainty_map.pow(2))
        loss_map = loss_map * mask
        loss = torch.sum(loss_map) / mask.sum()

        if mask is not None:
            valid_pixel = mask.sum() + 1
            diff_coord_map = mask * loss_reg

        thres_coord_map = torch.max(diff_coord_map - 0.05, torch.tensor([0.]).cuda())
        num_accurate = valid_pixel - thres_coord_map.nonzero().shape[0]
        accuracy = num_accurate / valid_pixel
        loss1 = torch.sum(loss_reg * mask) / mask.sum()
        return loss, accuracy, loss1


































