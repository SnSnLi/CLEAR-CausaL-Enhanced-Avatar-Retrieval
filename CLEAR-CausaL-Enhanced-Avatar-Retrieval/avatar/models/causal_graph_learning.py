import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

class StructuralLearning(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 结构学习层
        self.structure_learning = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # 转换特征
        img_hidden = self.feature_transform(image_features)
        text_hidden = self.feature_transform(text_features)
        
        # 计算所有可能的节点对
        batch_size = img_hidden.size(0)
        adj_matrix = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim)
        
        for i in range(self.hidden_dim):
            for j in range(self.hidden_dim):
                pair_features = torch.cat([img_hidden[:, i], text_hidden[:, j]], dim=-1)
                adj_matrix[:, i, j] = self.structure_learning(pair_features).squeeze()
                
        return adj_matrix

class CausalDiscovery(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 因果发现网络
        self.causal_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, structure: torch.Tensor) -> Tuple[torch.Tensor, nx.DiGraph]:
        batch_size = structure.size(0)
        causal_matrix = torch.zeros_like(structure)
        
        # 对每个批次进行因果发现
        for b in range(batch_size):
            # 转换为NetworkX图以便进行因果分析
            G = nx.from_numpy_array(structure[b].detach().numpy(), create_using=nx.DiGraph)
            
            # 应用因果发现算法
            for edge in G.edges():
                i, j = edge
                pair_features = torch.cat([
                    structure[b, i].unsqueeze(0),
                    structure[b, j].unsqueeze(0)
                ])
                causal_score = self.causal_net(pair_features)
                causal_matrix[b, i, j] = causal_score
                
            # 确保因果图的有向无环性
            causal_matrix[b] = self._ensure_dag(causal_matrix[b])
            
        return causal_matrix, G
    
    def _ensure_dag(self, matrix: torch.Tensor) -> torch.Tensor:
        """因果图是有向无环图"""
        adj_matrix = matrix.detach().numpy()
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        
        while not nx.is_directed_acyclic_graph(G):
            # 找到并移除循环
            cycles = list(nx.simple_cycles(G))
            if cycles:
                cycle = cycles[0]
                # 移除环中权重最小的边
                min_weight = float('inf')
                edge_to_remove = None
                
                for i in range(len(cycle)):
                    j = (i + 1) % len(cycle)
                    weight = adj_matrix[cycle[i]][cycle[j]]
                    if weight < min_weight:
                        min_weight = weight
                        edge_to_remove = (cycle[i], cycle[j])
                
                if edge_to_remove:
                    G.remove_edge(*edge_to_remove)
                    adj_matrix[edge_to_remove[0]][edge_to_remove[1]] = 0
                    
        return torch.from_numpy(adj_matrix)

class CausalGraphLearning(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 初始化子模块
        self.structural_learning = StructuralLearning(feature_dim, hidden_dim)
        self.causal_discovery = CausalDiscovery(hidden_dim)
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Dict:
        # 1. 结构学习
        structure = self.structural_learning(image_features, text_features)
        
        # 2. 因果发现
        causal_matrix, causal_graph = self.causal_discovery(structure)
        
        # 3. 验证因果不变性
        invariance_score = self.verify_causal_invariance(causal_matrix)
        
        return {
            'structure': structure,
            'causal_matrix': causal_matrix,
            'causal_graph': causal_graph,
            'invariance_score': invariance_score
        }
    
    def verify_causal_invariance(self, causal_matrix: torch.Tensor) -> torch.Tensor:
        """验证学习到的因果关系的不变性"""
        # 计算因果关系的稳定性分数
        stability_score = torch.mean(torch.abs(
            causal_matrix - torch.roll(causal_matrix, 1, dims=0)
        ))
        
        # 计算因果强度的一致性
        consistency_score = torch.std(causal_matrix, dim=0).mean()
        
        # 综合不变性分数
        invariance_score = 1.0 - (stability_score + consistency_score) / 2
        
        return invariance_score
    
    def get_causal_paths(self, causal_graph: nx.DiGraph) -> List[List[int]]:
        """获取因果图中的所有路径"""
        paths = []
        nodes = list(causal_graph.nodes())
        
        for source in nodes:
            for target in nodes:
                if source != target:
                    paths.extend(list(nx.all_simple_paths(causal_graph, source, target)))
                    
        return paths
    
    def extract_causal_mechanisms(self, causal_matrix: torch.Tensor) -> Dict[Tuple[int, int], float]:
        """提取因果机制的强度"""
        mechanisms = {}
        batch_size, n, _ = causal_matrix.shape
        
        # 计算平均因果强度
        avg_causal_matrix = torch.mean(causal_matrix, dim=0)
        
        for i in range(n):
            for j in range(n):
                if avg_causal_matrix[i, j] > 0:
                    mechanisms[(i, j)] = avg_causal_matrix[i, j].item()
                    
        return mechanisms