import torch
import torch.nn as nn
from typing import List, Dict, Tuple

def calculate_causal_loss(adj_matrix: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """
    计算因果损失
    Args:
        adj_matrix: 邻接矩阵 [N, N]
        features: 节点特征 [N, D]
    Returns:
        loss: 因果损失值
    """
    path_weights = compute_path_weights(adj_matrix, features)
    # 添加正则化项来鼓励稀疏的因果图
    sparsity_loss = torch.sum(torch.abs(adj_matrix))
    # 添加非循环约束
    acyclicity_loss = torch.sum(torch.matrix_power(adj_matrix, adj_matrix.size(0)))
    
    total_loss = path_weights.mean() + 0.1 * sparsity_loss + 0.1 * acyclicity_loss
    return total_loss

def create_adjacency_matrix(num_nodes: int) -> torch.Tensor:
    """
    创建邻接矩阵
    Args:
        num_nodes: 节点数量
    Returns:
        adj_matrix: [num_nodes, num_nodes]的上三角矩阵
    """
    adj_matrix = torch.triu(torch.ones(num_nodes, num_nodes), diagonal=1)
    return adj_matrix

def compute_path_weights(
    adj_matrix: torch.Tensor,
    features: torch.Tensor
) -> torch.Tensor:
    """
    计算因果路径权重
    Args:
        adj_matrix: 邻接矩阵 [N, N]
        features: 节点特征 [N, D]
    Returns:
        path_weights: 路径权重 [N, N]
    """
    path_weights = adj_matrix * torch.mm(features, features.t())
    return path_weights

def check_acyclicity(adj_matrix: torch.Tensor) -> bool:
    """
    检查图是否无环
    """
    n = adj_matrix.size(0)
    prod = torch.eye(n)
    for _ in range(n):
        prod = torch.mm(prod, adj_matrix)
        if torch.trace(prod) != 0:
            return False
    return True

def compute_causal_effect(
    adj_matrix: torch.Tensor,
    features: torch.Tensor,
    source: int,
    target: int
) -> float:
    """
    计算因果效应
    """
    path_weights = compute_path_weights(adj_matrix, features)
    direct_effect = path_weights[source, target]
    
    # 计算间接效应
    indirect_effect = 0
    n = adj_matrix.size(0)
    for k in range(n):
        if k != source and k != target:
            indirect_effect += path_weights[source, k] * path_weights[k, target]
    
    total_effect = direct_effect + indirect_effect
    return total_effect.item()