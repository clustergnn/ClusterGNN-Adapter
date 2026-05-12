from types import SimpleNamespace
import torch.nn as nn
import torch
from torch.nn import Sequential as Seq, Linear, ReLU, Parameter
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn.functional as F
import time
import pandas as pd
import os
from collections import defaultdict
import tsl
import math

class Model(MessagePassing):
    def __init__(self,configs):
        super(Model, self).__init__(aggr='add', node_dim=0) #  "Max" aggregation.
        self.num_clusters = configs.num_clusters
        self.dropout = configs.dropout
        
        improved=False
        add_self_loops=False
        normalize=True
        bias=True
        init_method='all'
                    
        
        source_nodes, target_nodes = [], []
        for i in range(self.num_clusters):
            for j in range(self.num_clusters):
                source_nodes.append(j)
                target_nodes.append(i)
                
        cluster_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        print("clustar edge:", cluster_edge_index.shape) # [2, num_clusters^2]
        self.register_buffer('cluster_edge_index', cluster_edge_index)
        
        
        # configs
        self.num_nodes = configs.enc_in
        self.in_channels = configs.seq_len
        self.out_channels = configs.seq_len
        
        
        # default
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.bias = bias
        self.init_method = init_method
        self.normalize = normalize
        
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        
        #######################
        ## ClusterGNN Layers ##
        #######################
        self.cluster_lin = torch.nn.Linear(self.in_channels, self.out_channels)
        self.cluster_norm = nn.LayerNorm(self.out_channels)
        self.cluster_proj = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.out_channels, self.out_channels)
        )
        self.out_norm = nn.LayerNorm(self.out_channels)
        
        ##################################
        ## FFN for message construction ##
        ##################################
        self.msg_mlp = nn.Sequential(
            nn.Linear(4 * self.out_channels, self.out_channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.out_channels, self.out_channels)
        )
        
        ####################
        ## FFN for gating ##
        ####################
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.out_channels, self.out_channels),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.out_channels, self.out_channels)
        )

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
       
        self._init_gumbel_logits_()
        self.reset_parameters()
        

    def _init_gumbel_logits_(self):
        # self.init_method: 'all' / 'random' / 'equal'

        # 1. Node-to-cluster assignment logits: [num_nodes, num_clusters]
        if self.init_method == 'all':
            logits_cluster = 0.8 * torch.ones(self.num_nodes, self.num_clusters)
            #logits_cluster = 1e-3 * torch.randn(self.num_nodes, self.num_clusters)
        elif self.init_method == 'random':
            logits_cluster = 1e-3 * torch.randn(self.num_nodes, self.num_clusters)
        elif self.init_method == 'equal':
            logits_cluster = 0.5 * torch.ones(self.num_nodes, self.num_clusters)
        else:
            raise NotImplementedError(f'Init method "{self.init_method}" not supported for logits_cluster')
        
        self.register_parameter('logits_cluster', Parameter(logits_cluster, requires_grad=True))
        
        
        # 2. Edge existence logits between clusters: [num_clusters^2, 2]
        num_cluster_edges = self.num_clusters ** 2
        if self.init_method == 'all':
            #logits_edge = 1e-3 * torch.randn(num_cluster_edges, 2)
            logits_edge = 0.8 * torch.ones(num_cluster_edges, 2)
            logits_edge[:, 1] = 0.0  # "엣지 있음"의 초기 확률 낮춤
        elif self.init_method == 'random':
            logits_edge = 1e-3 * torch.randn(num_cluster_edges, 2)
        elif self.init_method == 'equal':
            logits_edge = 0.5 * torch.ones(num_cluster_edges, 2)
        else:
            raise NotImplementedError(f'Init method "{self.init_method}" not supported for logits_edge')
        print("logits_edge:", logits_edge.shape) # [num_clusters², 2]
        self.register_parameter('logits_edge', Parameter(logits_edge, requires_grad=True))
    
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        
    

    def forward(self, x):
        
        # x: MTF standard format(bsz, seq_len, num_nodes)
        
        x = x.permute(0, 2, 1) # >> (bsz, num_nodes, seq_len)
        x = x.transpose(0, 1) # >> (num_nodes, bsz, seq_len)
        
        # --------------------------------------------------------------
        # 1) Cluster assignment (Gumbel)
        # --------------------------------------------------------------
                

        logits_clean = torch.clamp(
            torch.nan_to_num(self.logits_cluster, nan=0.0, posinf=10.0, neginf=-10.0), # Nan만 0으로, 양의 무한대는 20으로, 음의 무한대는 -20으로 클램핑
            -10, 10
        )
        S = F.gumbel_softmax(logits_clean, tau=0.5, hard=False)   # [N, K]

        # --------------------------------------------------------------
        # 2) Cluster feature construction (node average pooling)
        # --------------------------------------------------------------
        cluster_mass = S.sum(dim=0).clamp_min(1e-6)            # [K]
        X_cluster = torch.einsum('nk,nbf->kbf', S, x)          # [K, B, F]
        X_cluster = X_cluster / cluster_mass[:, None, None]

        X_cluster = self.cluster_lin(X_cluster)
        X_cluster = self.cluster_norm(X_cluster)
        X_cluster = F.gelu(X_cluster)
        

        # --------------------------------------------------------------
        # 3) Inter-cluster edge selection (Gumbel)
        # --------------------------------------------------------------
       
        logits_edge_clean = torch.clamp(
            torch.nan_to_num(self.logits_edge, nan=0.0, posinf=10.0, neginf=-10.0),
                -10, 10
        )
        z_edge = F.gumbel_softmax(logits_edge_clean, tau=0.5, hard=False)


        # --------------------------------------------------------------
        # 4) Cluster graph normalization
        # --------------------------------------------------------------
        edge_weight = None
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                self.cluster_edge_index,
                edge_weight,
                self.num_clusters,
                self.improved,
                self.add_self_loops,
                dtype=x.dtype
            )
        else:
            edge_index, edge_weight = self.cluster_edge_index, None

        # --------------------------------------------------------------
        # 5) Cluster graph message passing
        # --------------------------------------------------------------
        out_cluster = self.propagate(
            edge_index,
            x=X_cluster,
            edge_weight=edge_weight,
            size=None,
            z=z_edge
        )   # [K, B, F]

        
        # --------------------------------------------------------------
        # 6) Cluster-to-node lifting
        # --------------------------------------------------------------
        cluster_vector = torch.einsum('nk,kbf->nbf', S, out_cluster)

        cluster_vector = self.cluster_proj(cluster_vector)
        cluster_vector = self.out_norm(cluster_vector)
        cluster_vector = F.gelu(cluster_vector)
        cluster_vector = F.dropout(cluster_vector, p=0.4, training=self.training)

        # --------------------------------------------------------------
        # 7) fusion
        # --------------------------------------------------------------
        fusion_input = torch.cat([x, cluster_vector], dim=-1)
        self.gate = torch.sigmoid(self.gate_mlp(fusion_input))
        gated_cluster = self.gate * cluster_vector
        out = x + gated_cluster #[N, B, F]


        if self.bias is not None:
            out = out + self.bias
            
        
        # logging용
        with torch.no_grad():
            self.last_gate_per_channel = self.gate.detach().mean(dim=(1, 2)).cpu()
            
        # cleanup
        del X_cluster, out_cluster, cluster_vector, z_edge
        self.cluster_x_repr = out
        
        # --------------------------------------------------------------
        # Restore original format
        # --------------------------------------------------------------
        out = out.transpose(0, 1)   # [B, N, L]
        out = out.permute(0, 2, 1)  # [B, L, N]
        
        # backbone-compatible output format: [B, L, N]
        return out

    def message(self, x_i, x_j, edge_weight, z):
        # x_i, x_j: [E, B, F]
        diff = x_j - x_i
        summ = x_j + x_i

        # [E, B, 4F]
        msg_input = torch.cat([x_i, x_j, diff, summ], dim=-1)

        #[K, B, F]
        msg = self.msg_mlp(msg_input)

        # z: [E, 2] 라고 가정
        gate = z[:, 1].view(-1, 1, 1)

        if edge_weight is not None:
            weight = edge_weight.view(-1, 1, 1)
            msg = msg * weight

        return msg * gate
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
 