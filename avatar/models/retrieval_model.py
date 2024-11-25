import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple
from .causal_graph_learning import CausalGraphLearning
from .attention import CausalAttention

class CausalMatchingModule(nn.Module):
    """因果匹配模块"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 因果强度评估网络
        self.causal_strength_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        # 因果路径聚合网络
        self.path_aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def evaluate_causal_path(self, features: torch.Tensor, path: List[int]) -> torch.Tensor:
        """评估单个因果路径的强度"""
        path_features = []
        for i in range(len(path)-1):
            src, dst = path[i], path[i+1]
            pair_feature = torch.cat([features[:, src], features[:, dst]], dim=-1)
            strength = self.causal_strength_net(pair_feature)
            path_features.append(strength)
            
        # 聚合路径上的所有因果强度
        path_features = torch.stack(path_features, dim=1)
        path_strength = self.path_aggregation(path_features)
        
        return path_strength
    
    def forward(self, 
                img_features: torch.Tensor, 
                text_features: torch.Tensor,
                causal_graph: nx.DiGraph) -> torch.Tensor:
        """计算因果匹配分数"""
        batch_size = img_features.size(0)
        
        # 获取所有因果路径
        all_paths = list(nx.all_simple_paths(causal_graph, source=0, target=len(causal_graph.nodes())-1))
        
        path_strengths = []
        # 评估每条因果路径
        for path in all_paths:
            # 计算图像特征的路径强度
            img_path_strength = self.evaluate_causal_path(img_features, path)
            # 计算文本特征的路径强度
            text_path_strength = self.evaluate_causal_path(text_features, path)
            # 综合路径强度
            path_strength = (img_path_strength * text_path_strength).sqrt()
            path_strengths.append(path_strength)
            
        # 聚合所有路径的匹配分数
        if path_strengths:
            matching_scores = torch.stack(path_strengths, dim=1)
            final_scores = torch.max(matching_scores, dim=1)[0]
        else:
            final_scores = torch.zeros(batch_size, 1).to(img_features.device)
            
        return final_scores

class RetrievalModel(nn.Module):
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 output_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 因果图学习模块
        self.causal_graph_learner = CausalGraphLearning(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )
        
        # 因果注意力模块
        self.causal_attention = CausalAttention(
            hidden_dim=hidden_dim
        )
        
        # 因果匹配模块
        self.causal_matching = CausalMatchingModule(
            hidden_dim=hidden_dim
        )
        
        # 特征投影网络
        self.image_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def extract_causal_features(self, 
                              img_features: torch.Tensor, 
                              text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取因果增强的特征"""
        # 1. 学习因果图
        causal_info = self.causal_graph_learner(img_features, text_features)
        causal_graph = causal_info['causal_graph']
        
        # 2. 应用因果注意力
        attended_features = self.causal_attention(
            img_features=img_features,
            text_features=text_features,
            causal_graph=causal_graph
        )
        
        return {
            'image': attended_features['image'],
            'text': attended_features['text'],
            'causal_graph': causal_graph,
            'causal_info': causal_info
        }
    
    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Dict:
        # 1. 提取因果特征
        causal_features = self.extract_causal_features(images, texts)
        
        # 2. 投影到特征空间
        img_features = self.image_projector(causal_features['image'])
        text_features = self.text_projector(causal_features['text'])
        
        # 3. 计算因果匹配分数
        matching_scores = self.causal_matching(
            img_features=img_features,
            text_features=text_features,
            causal_graph=causal_features['causal_graph']
        )
        
        return {
            'matching_scores': matching_scores,
            'img_features': img_features,
            'text_features': text_features,
            'causal_info': causal_features['causal_info']
        }
    
    def get_retrieval_results(self, matching_scores: torch.Tensor, k: int = 5) -> Dict:
        """获取检索结果"""
        values, indices = matching_scores.topk(k, dim=-1)
        return {
            'scores': values,
            'indices': indices
        }
        
    def compute_loss(self, matching_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算因果匹配损失"""
        # 使用二元交叉熵损失
        loss = nn.BCELoss()(matching_scores, labels)
        
        # 可以添加其他正则化项，如因果稀疏性maybe……再说吧
        # causal_sparsity = self.compute_causal_sparsity()
        # loss += lambda_sparsity * causal_sparsity
        
        return loss
