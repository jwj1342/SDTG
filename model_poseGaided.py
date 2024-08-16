import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import init
from torch_geometric.nn import GATConv,global_mean_pool,global_max_pool
from torch_geometric.data import Data, Batch
from resnet import resnet50
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x



def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization
    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class gcn_resnet(nn.Module):
    def __init__(self, arch='resnet50', return_feature_maps = False):
        super(gcn_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.return_feature_maps = return_feature_maps

    def forward(self, x, task="training"):
        x1 = self.base.layer1(x)  # [256, 72, 36]     
        x2 = self.base.layer2(x1)  # [512, 36, 18]   
        x3 = self.base.layer3(x2)  # [1024, 18, 9]
        x4 = self.base.layer4(x3)  # [2048, 18, 9]

        if self.return_feature_maps:
            return x4, x  
    
        if task == "training":
            dis_x2 = self.same_modal_disturbance(x2)
            dis_x3 = self.base.layer3(dis_x2)
            dis_x3 = self.cross_modal_disturbance(dis_x3)
            x_dis = self.base.layer4(dis_x3)
            return x4, x_dis
        else:
            return x4

    def same_modal_disturbance(self, x):
        B = x.size(0)
        x_v = x[:B // 2]
        x_t = x[B // 2:]
        noise_v = x_v[torch.randperm(x_v.size(0))]  # randomly select a turbulent frame
        noise_t = x_t[torch.randperm(x_t.size(0))]
        ##############################################
        distur_v = adain(x_v, noise_v)
        distur_t = adain(x_t, noise_t)
        distur_x = torch.cat((distur_v, distur_t), dim=0)
        return distur_x

    def cross_modal_disturbance(self, x):

        B = x.size(0)
        x_v = x[:B // 2]
        x_t = x[B // 2:]
        noise_v = x_v[torch.randperm(x_v.size(0))]
        noise_t = x_t[torch.randperm(x_t.size(0))]
        distur_v = adain(x_v, noise_t)
        distur_t = adain(x_t, noise_v)
        distur_x = torch.cat((distur_v, distur_t), dim=0)

        return distur_x


class SimplifiedGATBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3, heads=4):
        super(SimplifiedGATBlock, self).__init__()
        # Reduced number of heads and layers, adjusted hidden channels
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        # Only one hidden layer, reduce the feature size increase and heads
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        # Pass through the first convolution
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)  # adjusted dropout probability

        # Pass through the second convolution
        x = self.conv2(x, edge_index)
        x = F.elu(x)  # Final activation

        # Global mean pooling to aggregate all node features into a single graph-level representation
        # x = global_mean_pool(x, batch)  # batch is the batch vector, which assigns each node to a specific graph

        return x


def calculate_edge_features(pose, connections):
    vectors = pose[:, connections[1], :2] - pose[:, connections[0], :2]  # Get vectors between joints
    lengths = vectors.norm(dim=-1).unsqueeze(-1)  # Calculate lengths
    angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0]).unsqueeze(-1)  # Calculate angles
    edge_features = torch.cat([lengths, angles], dim=-1)  # Concatenate length and angle features
    return edge_features


class FrameLevelAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FrameLevelAttention, self).__init__()
        
        # 交叉注意力模块用于在模态之间进行信息交换
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        # 添加反向的交叉注意力模块
        self.reverse_cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)

        self.bn_enhanced_origin = nn.BatchNorm1d(feature_dim)
        self.bn_enhanced_time_guide = nn.BatchNorm1d(feature_dim)  # 新增加的BN层
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )
    def forward(self, time_guide, origin):
        
        # 原始方向的注意力：从 time_guide 到 origin
        enhanced_origin, _ = self.cross_attn(time_guide, origin, origin)
        enhanced_time, _ = self.reverse_cross_attn (origin, time_guide, time_guide)
        
        # 应用ReLU激活函数确保非负输出
        time_guided_fea = self.ffn(torch.concat((enhanced_origin, enhanced_time),dim=-1)) + origin + 1e-3 
        
        return time_guided_fea

class GuidedGeMPooling(nn.Module):
    def __init__(self, num_p, feature_dim):
        super(GuidedGeMPooling, self).__init__()
        self.num_p = num_p
        self.fc_layers = nn.ModuleList([nn.Linear(2*feature_dim, 2048) for _ in range(num_p)])

    def forward(self, x, p_list, guide_vector):
        batch_size, seq_length, feature_dim = x.shape
        pooled_features = []

        for p in p_list:
            pooled = (torch.mean(x ** p, dim=1) + 1e-12) ** (1 / p)
            pooled_features.append(pooled)

        stacked_features = torch.stack(pooled_features, dim=1)

        attention_scores = torch.stack([self.fc_layers[i](guide_vector) for i in range(self.num_p)], dim=1)

        attention_weights = F.softmax(attention_scores, dim=1)

        weighted_features = stacked_features * attention_weights
        aggregated_features = torch.sum(weighted_features, dim=1)

        return aggregated_features
    

class embed_net(nn.Module):
    def __init__(self, class_num, drop=0.2, arch="resnet50", return_feature_maps=False):
        super(embed_net, self).__init__()

        # hyper parameters
        pool_dim = 2048
        num_nodes=17
        seq_length=12
        self.dropout = drop
        
        # feature extract
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = gcn_resnet(arch=arch, return_feature_maps=return_feature_maps)

        # classification layers
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.local_bottleneck = nn.BatchNorm1d(pool_dim)
        self.local_bottleneck.bias.requires_grad_(False)  # no shift
        self.local_classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.global_bottleneck = nn.BatchNorm1d(pool_dim)
        self.global_bottleneck.bias.requires_grad_(False)  # no shift
        self.global_classifier = nn.Linear(pool_dim, class_num, bias=False)
        

        # initialize
        self.local_bottleneck.apply(weights_init_kaiming)
        self.local_classifier.apply(weights_init_classifier)
        self.global_bottleneck.apply(weights_init_kaiming)
        self.global_classifier.apply(weights_init_classifier)

        self.gat = SimplifiedGATBlock(in_channels=3, out_channels= 12, hidden_channels=16, heads=2, dropout=0.6)
        
        self.bn = nn.BatchNorm1d(12)
        self.bn.bias.requires_grad_(False)
        self.lstm_edge = nn.LSTM(64, 1024, 1, batch_first=True, bidirectional=True)
        self.time_guide = FrameLevelAttention(pool_dim)
        connections = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6],
            [5, 7], [6, 8],
            [7, 9], [8, 10],
            [1, 2],
            [0, 1], [0, 2],
            [1, 3], [2, 4],
            [3, 5], [4, 6]
        ]
        self.connections = torch.tensor(connections, dtype=torch.long).t().contiguous().cuda()
        
        temporal_connections = [[i, i + num_nodes] for i in range(num_nodes * (seq_length - 1))]
        connections += temporal_connections

        connections += [[b, a] for a, b in connections]
        self.connections1 = torch.tensor(connections, dtype=torch.long).t().contiguous().cuda()

        self.gat_avgpool = nn.AdaptiveAvgPool1d(2048)
        
        self.GuidedAggModule = GuidedGeMPooling(num_p = 5, feature_dim = 2048)
        
        self.fc1 = nn.Linear(38,64)
        self.pose_classifier = nn.Linear(2*pool_dim, class_num)
        self.pose_bottleneck = nn.BatchNorm1d(2*pool_dim)
    
    def GAT(self, pose):
        batch_size, seq_length, num_nodes, _ = pose.size()

        data_list = []
        gat_input = pose.view(batch_size, seq_length * num_nodes, -1)

        for i in range(batch_size):
            data = Data(x=gat_input[i], edge_index=self.connections1)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        gat_output = self.gat(batch.x, batch.edge_index).view(batch_size,-1)
        
        gat_output = self.gat_avgpool(gat_output)
        return gat_output

    def forward(self, x1, x2, p1, p2, modal=0, seq_len=6):
        
        b, c, h, w = x1.size()
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)

        # style augmentation
        if self.training:
            # IR modality

            frame_batch = seq_len * b
            delta = torch.rand(frame_batch) + 0.5 * torch.ones(frame_batch)  # [0.5-1.5]
            inter_map = delta.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1).cuda()
            x2 = x2 * inter_map

            # RGB modality
            alpha = (torch.rand(frame_batch) + 0.5 * torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(
                dim=1).unsqueeze(dim=1)
            beta = (torch.rand(frame_batch) + 0.5 * torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(
                dim=1).unsqueeze(dim=1)
            gamma = (torch.rand(frame_batch) + 0.5 * torch.ones(frame_batch)).unsqueeze(dim=1).unsqueeze(
                dim=1).unsqueeze(dim=1)
            inter_map = torch.cat((alpha, beta, gamma), dim=1).cuda()
            x1 = x1 * inter_map
            for i in range(x1.shape[0]):
                x1[i] = x1[i, torch.randperm(3), :, :]

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)

            pose = torch.cat((p1, p2), 0)
            gat_outputs = self.GAT(pose)
            edge_features = calculate_edge_features(pose.view(-1, 17, 3), self.connections).view(b*2,seq_len,-1)
            
        elif modal == 1:
            x = self.visible_module(x1)

            gat_outputs = self.GAT(p1)
            edge_features = calculate_edge_features(p1.view(-1, 17, 3), self.connections).view(b,seq_len,-1)
            
        elif modal == 2:
            x = self.thermal_module(x2)

            gat_outputs = self.GAT(p2)
            edge_features = calculate_edge_features(p2.view(-1, 17, 3), self.connections).view(b,seq_len,-1)
            
        
        if self.training:
            x, x_dis = self.base_resnet(x)
        else:
            x = self.base_resnet(x, task="testing")
        
        
        edge_features = self.fc1(edge_features)
        edge_output, _ = self.lstm_edge(edge_features)
        
        pose_feat = torch.cat((edge_output[:,-1,:], gat_outputs), dim=-1)
        
        if self.training:

            x_local = self.avgpool(x).squeeze()
            x_local_2 = rearrange(x_local, '(b t) n->b t n', t=seq_len)
            
            x_local_2 = self.time_guide(edge_output, x_local_2)
            
            x_dis_local = self.avgpool(x_dis).squeeze()
            x_dis_local_2 = rearrange(x_dis_local, '(b t) n->b t n', t=seq_len)
            # x_dis_local_2 = self.time_guide(edge_output, x_dis_local_2)
            
            x_local_feat = self.local_bottleneck(x_local)
            x_local_logits = self.local_classifier(x_local_feat)

            x_dis_feat = self.local_bottleneck(x_dis_local)
            x_dis_logits = self.local_classifier(x_dis_feat)

            # p = 3.0
            # x_global = (torch.mean(x_local_2 ** p, dim=1) + 1e-12) ** (1 / p)
            pose_feat = self.pose_bottleneck(pose_feat)
            x_global = self.GuidedAggModule(x_local_2, [0.7, 1.0, 3.0, 5.0, 7.0], pose_feat)

            global_feat = self.global_bottleneck(x_global)
            logits = self.global_classifier(global_feat)

            defense_loss = torch.mean(torch.sqrt((x_local - x_dis_local).pow(2).sum(1)))
            
            logits_pose = self.pose_classifier(pose_feat)
            # log_a =F.log_softmax(gat_outputs)
            # softmax_b =F.softmax(edge_output,dim=-1)
            # kl_mean = F.kl_div(log_a, softmax_b, reduction='mean')
            return x_global, x_local, logits, x_local_logits, x_dis_logits, defense_loss, logits_pose
        else:

            x_local = self.avgpool(x).squeeze()
            x_local_2 = rearrange(x_local, '(b t) n->b t n', t=seq_len)
            x_local_2 = self.time_guide(edge_output, x_local_2)
            # p = 3.0
            # x_global = (torch.mean(x_local_2 ** p, dim=1) + 1e-12) ** (1 / p)
            pose_feat = self.pose_bottleneck(pose_feat)
            x_global = self.GuidedAggModule(x_local_2, [0.7, 1.0, 3.0, 5.0, 7.0], pose_feat)
            
            global_feat = self.global_bottleneck(x_global)
            # global_feat = self.cross_modal_attention(pose_feat, global_feat) + global_feat
            return self.l2norm(global_feat)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input1 = torch.randn(2, 36, 288, 144).to(device)
    input2 = torch.randn(2, 36, 288, 144).to(device)
    input3 = torch.randn(2, 12, 17, 3).to(device)
    input4 = torch.randn(2, 12, 17, 3).to(device)
    net = embed_net(class_num=500, drop=0.2, arch="resnet50").to(device)
    x_global, x_local, logits, x_local_logits, x_dis_logits, defense_loss, logits_pose = net(input1, input2, input3, input4, modal=0, seq_len=12)
    print('-----------------------------------')
    print(x_global.shape)
    print(x_local.shape)
    print(logits.shape)
    print(x_local_logits.shape)
    print(x_dis_logits.shape)
    print(defense_loss)
    print("Model train has been tested successfully!")

    print('-----------------------------------')
    net.eval()
    global_feat1 = net(input1, input2, input3, input4, modal=1, seq_len=12)
    global_feat2 = net(input1, input2, input3, input4, modal=2, seq_len=12)
    print(global_feat1.shape)
    print(global_feat2.shape)
    print("Model eval has been tested successfully!")
    print('-----------------------------------')