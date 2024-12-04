import torch
import torch.nn as nn
import networkx as nx
from typing import Dict

class CausalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 因果注意力机制
        self.causal_attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                img_features: torch.Tensor, 
                text_features: torch.Tensor,
                causal_graph: nx.DiGraph) -> Dict[str, torch.Tensor]:
        """应用因果注意力
        Args:
            img_features: 图像特征 [batch_size, img_dim]
            text_features: 文本特征 [batch_size, text_dim]
            causal_graph: 因果图结构
            
        Returns:
            Dict包含:
                'image': 注意力加权后的图像特征
                'text': 注意力加权后的文本特征
        """
        batch_size, img_dim = img_features.size()
        _, text_dim = text_features.size()
        
        # 创建因果注意力矩阵
        causal_attention_matrix = torch.zeros(batch_size, img_dim, text_dim).to(img_features.device)
        
        # 遍历所有因果路径
        for source, target in causal_graph.edges():
            # 计算因果注意力权重
            causal_attention = self.causal_attention_net(
                torch.cat([img_features[:, source], text_features[:, target]], dim=-1)
            )
            
            # 更新因果注意力矩阵
            causal_attention_matrix[:, source, target] = causal_attention
            
        # 应用因果注意力
        attended_img_features = torch.matmul(causal_attention_matrix, text_features)
        attended_text_features = torch.matmul(causal_attention_matrix.transpose(1, 2), img_features)
        
        return {
            'image': attended_img_features,
            'text': attended_text_features
        }