import cv2
import numpy as np
import math
from sklearn.decomposition import PCA
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PositionEncodingSine(nn.Module):
    def __init__(self, d_model, max_shape=(60, 80)):
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)
        self.pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe.unsqueeze(0), persistent=False)                     # [1, C, H, W]

    # [B C H W]
    def forward(self, x):
        return self.pe[:, :, :x.size(2), :x.size(3)]

# [B N C]
def PCA_Analysis(feat_c0, num_component=1):
    b = feat_c0.shape[0]
    pca = PCA(n_components=num_component)
    pca_res = []
    for i in range(b):
        new_feat_c0_show = pca.fit_transform(feat_c0[i].permute(1,0)).transpose(1,0)      # [M C]
        pca_res.append(new_feat_c0_show)
    pca_res = torch.tensor(pca_res)

if __name__ == "__main__":
    pos_enc = PositionEncodingSine(64, max_shape=(240,320))
    x = torch.rand(4,64,240,320)
    pos_enc_out = pos_enc(x)                # [1 64 240 320]
    plt.imshow(torch.sum(pos_enc_out[0], dim=0).cpu().numpy(), cmap="jet")
    plt.show()































