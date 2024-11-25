import torch
import numpy as np
from typing import List, Tuple
from typing import Dict

def compute_ranks(
    image_features: torch.Tensor,
    text_features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算检索排名
    Args:
        image_features: 图像特征 [N, D]
        text_features: 文本特征 [N, D]
    Returns:
        i2t_ranks: 图搜文排名
        t2i_ranks: 文搜图排名
    """
    # 计算相似度矩阵
    similarity = image_features @ text_features.t()
    
    # 计算排名
    i2t_ranks = []
    t2i_ranks = []
    
    for i in range(similarity.shape[0]):
        # 图搜文
        i2t_sim = similarity[i]
        inds = torch.argsort(i2t_sim, descending=True)
        rank = torch.where(inds == i)[0][0]
        i2t_ranks.append(rank.item())
        
        # 文搜图
        t2i_sim = similarity[:, i]
        inds = torch.argsort(t2i_sim, descending=True)
        rank = torch.where(inds == i)[0][0]
        t2i_ranks.append(rank.item())
    
    return torch.tensor(i2t_ranks), torch.tensor(t2i_ranks)

def compute_metrics(
    i2t_ranks: torch.Tensor,
    t2i_ranks: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    计算R@K和NDCG@K指标
    """
    metrics = {}
    
    # 计算R@K
    for k in k_values:
        i2t_r_k = (i2t_ranks < k).float().mean().item()
        t2i_r_k = (t2i_ranks < k).float().mean().item()
        metrics[f'i2t_R@{k}'] = i2t_r_k * 100
        metrics[f't2i_R@{k}'] = t2i_r_k * 100
    
    # 计算NDCG@K
    for k in k_values:
        i2t_ndcg = compute_ndcg(i2t_ranks, k)
        t2i_ndcg = compute_ndcg(t2i_ranks, k)
        metrics[f'i2t_NDCG@{k}'] = i2t_ndcg * 100
        metrics[f't2i_NDCG@{k}'] = t2i_ndcg * 100
    
    return metrics

def compute_ndcg(ranks: torch.Tensor, k: int) -> float:
    """计算NDCG@K"""
    dcg = torch.zeros_like(ranks, dtype=torch.float)
    for i, rank in enumerate(ranks):
        if rank < k:
            dcg[i] = 1.0 / torch.log2(rank + 2)
    
    ideal_ranks = torch.zeros_like(ranks)
    idcg = torch.zeros_like(ranks, dtype=torch.float)
    for i in range(min(k, len(ranks))):
        idcg[i] = 1.0 / torch.log2(i + 2)
    
    return (dcg.sum() / idcg.sum()).item()