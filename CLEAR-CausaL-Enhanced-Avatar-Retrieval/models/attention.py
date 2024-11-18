# attention.py

import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, Tuple

class CausalWeighting(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 因果权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, causal_graph: nx.DiGraph) -> torch.Tensor:
        batch_size = features.size(0)
        num_nodes = features.size(1)
        
        # 初始化权重矩阵
        weights = torch.zeros(batch_size, num_nodes, num_nodes).to(features.device)
        
        # 基于因果图计算权重
        for i in range(num_nodes):
            for j in range(num_nodes):
                if causal_graph.has_edge(i, j):
                    pair_features = torch.cat([
                        features[:, i],
                        features[:, j]
                    ], dim=-1)
                    weights[:, i, j] = self.weight_net(pair_features).squeeze()
                    
        return weights

class CausalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 初始化因果权重模块
        self.causal_weight = CausalWeighting(hidden_dim)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # 因果一致性检查器
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def apply_causal_attention(self, 
                             features: torch.Tensor,
                             causal_weights: torch.Tensor) -> torch.Tensor:
        """应用因果引导的注意力"""
        # 将因果权重应用到注意力计算中
        attn_output, _ = self.multihead_attn(
            query=features,
            key=features,
            value=features,
            attn_mask=causal_weights
        )
        
        return attn_output
    
    def check_causal_consistency(self, attended_features: torch.Tensor) -> torch.Tensor:
        """检查因果一致性"""
        consistency_scores = self.consistency_checker(attended_features)
        return consistency_scores
    
    def forward(self, 
                img_features: torch.Tensor,
                text_features: torch.Tensor,
                causal_graph: nx.DiGraph) -> Dict[str, torch.Tensor]:
        # 1. 计算因果权重
        img_weights = self.causal_weight(img_features, causal_graph)
        text_weights = self.causal_weight(text_features, causal_graph)
        
        # 2. 应用因果注意力
        attended_img = self.apply_causal_attention(img_features, img_weights)
        attended_text = self.apply_causal_attention(text_features, text_weights)
        
        # 3. 检查因果一致性
        img_consistency = self.check_causal_consistency(attended_img)
        text_consistency = self.check_causal_consistency(attended_text)
        
        # 4. 跨模态因果注意力
        cross_modal_features = self.apply_cross_modal_attention(attended_img, attended_text)
        
        return {
            'image': attended_img,
            'text': attended_text,
            'cross_modal': cross_modal_features,
            'causal_info': {
                'img_weights': img_weights,
                'text_weights': text_weights,
                'img_consistency': img_consistency,
                'text_consistency': text_consistency
            }
        }
    
    def apply_cross_modal_attention(self, 
                                  img_features: torch.Tensor,
                                  text_features: torch.Tensor) -> torch.Tensor:
        """应用跨模态因果注意力"""
        # 计算跨模态注意力
        cross_attn_output, _ = self.multihead_attn(
            query=img_features,
            key=text_features,
            value=text_features
        )
        
        return cross_attn_output