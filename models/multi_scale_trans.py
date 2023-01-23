import os
import sys
sys.path.append(".")
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .resnet_backbone import resnet18
from .utils import PositionEncodingSine

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class SENet(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, padding=1, kernel_size=3):
        super(SENet, self).__init__()
        self.conv = conv(in_chan, out_chan, stride=stride, padding=padding, kernel_size=kernel_size)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

class Struct_Info(nn.Module):
    def __init__(self, k):
        super(Struct_Info, self).__init__()
        self.embed_conv = SENet(64, 16, stride=16, padding=0, kernel_size=16)
        self.k = k
        self.change_conv = nn.Linear(16,16)
        self.pos_embed2 = PositionEncodingSine(16, max_shape=(60, 80))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def KNN(self, feat):
        B,H,W,C = feat.shape
        N = H * W
        feat = feat.reshape(B,N,C)
        dist = torch.norm(feat[:,:,None,:] - feat[:,None,:,:], p=2, dim=-1, keepdim=False)
        # dist[torch.arange(B)[:,None].repeat(1, N),
        #      torch.arange(N)[None, :].repeat(B, 1),
        #      torch.arange(N)[None, :].repeat(B, 1)] = 10000

        interval = N / self.k
        knn_array = torch.arange(interval/2, N, interval).long()
        topk_val, topk_index = torch.topk(dist, dim=-1, k=N)
        knn_index = topk_index[:,:,knn_array]

        b_index = torch.arange(B)[:,None,None].repeat(1, N, self.k)
        knn_feat = feat[b_index, knn_index, :]
        feat_ = feat[:,:,None,:].repeat(1, 1, self.k, 1)
        knn_edge_feat = knn_feat - feat_
        knn_edge_feat = knn_edge_feat.reshape(B*N, self.k, -1)
        knn_edge_feat = self.change_conv(knn_edge_feat)

        knn_edge_feat = torch.mean(knn_edge_feat, dim=-2).reshape(B,N,-1)
        knn_edge_feat = knn_edge_feat.permute(0,-1,1).reshape(B,-1,H,W)
        return knn_edge_feat

    def forward(self, x):
        embed = self.embed_conv(x)
        B,C,H,W = embed.shape
        embed = embed.permute(0,2,3,1)
        knn_edge_feat = self.KNN(embed)

        knn_edge_feat = self.up(knn_edge_feat)
        knn_edge_feat = self.up(knn_edge_feat)
        knn_edge_feat = knn_edge_feat + self.pos_embed2(knn_edge_feat).cuda()
        return knn_edge_feat


class Multi_Scale_Trans(nn.Module):
    def __init__(self):
        super(Multi_Scale_Trans, self).__init__()
        cnn = resnet18(pretrained=True)
        self.cnn_pre = nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu)
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4

        self.SE2 = SENet(in_chan=128, out_chan=256, stride=2)
        self.SE3 = SENet(in_chan=256, out_chan=256, stride=1)
        self.SE4 = SENet(in_chan=512, out_chan=256, stride=1)
        self.conv3_4 = conv(256, 256)
        self.conv2_3 = conv(256, 256)

        self.last_conv = nn.Sequential(
            conv(768 + 16, 512),
            conv(512, 256),
            conv(256, 128)
        )
        self.coord_regress = nn.Sequential(
            conv(128, 64),
            nn.Conv2d(64, 3, kernel_size=1, padding=0)
        )
        self.uncer_regress = nn.Sequential(
            conv(128, 64),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        B,C,H,W = 4, 64, 240, 320
        self.pos_embed = PositionEncodingSine(C, max_shape=(H, W))
        self.struct_info = Struct_Info(k=60)

    def forward(self, x):
        out = self.cnn_pre(x)
        out = out + self.pos_embed(out).cuda()

        out_layer1 = self.layer1(out)
        out_layer2 = self.layer2(out_layer1)
        out_layer3 = self.layer3(out_layer2)
        out_layer4 = self.layer4(out_layer3)

        out_layer2_1 = self.SE2(out_layer2)
        out_layer3_1 = self.SE3(out_layer3)
        out_layer4_1 = self.SE4(out_layer4)
        edge_feats = self.struct_info(out_layer1)

        out_layer3_4 = self.conv3_4(out_layer3_1 + out_layer4_1)
        weight4_3 = torch.sigmoid(out_layer3_4)
        weight4 = 1 - weight4_3
        out_layer2_3 = self.conv2_3(out_layer2_1 + out_layer3_1)
        weight2_3 = torch.sigmoid(out_layer2_3)
        weight2 = 1 - weight2_3
        weight3 = (weight4_3 + weight2_3) / 2

        out_layer2_final = out_layer2_1 * weight2
        out_layer3_final = out_layer3_1 * weight3
        out_layer4_final = out_layer4_1 * weight4

        out = out_layer2_final + out_layer3_final + out_layer4_final
        out = torch.cat((out_layer4, out, edge_feats), dim=1)

        out = self.last_conv(out)
        coord = self.coord_regress(out)
        uncer = self.uncer_regress(out)
        uncer = torch.sigmoid(uncer)

        return coord, uncer


if __name__ == "__main__":
    x = torch.randn(4,3,480,640)
    net = Multi_Scale_Trans()
    coord, conf = net(x)
    print(coord.shape)
    print(conf.shape)














