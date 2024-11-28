import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Tuple, Optional
from causal_graph_definition import CausalGraphDefinition

class CausalEnhancedDiscovery(nn.Module):
    """增强版因果发现模块，整合预定义因果图与CLEAR项目"""
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        causal_dim: int = 256,
        num_causal_vars: int = 32,
        prior_weight: float = 0.3,  # 预定义因果图的权重
        temperature: float = 0.1
    ):
        super().__init__()
        
        # 导入预定义因果图
        self.causal_graph_def = CausalGraphDefinition()
        self.prior_graph = self.causal_graph_def.get_prior_graph()
        self.prior_weight = prior_weight
        
        # 基础组件
        self.modality_dims = modality_dims
        self.causal_dim = causal_dim
        self.num_causal_vars = num_causal_vars
        
        # CLEAR项目兼容的编码器
        self.modal_encoders = nn.ModuleDict({
            modality: self._build_clear_encoder(dim) 
            for modality, dim in modality_dims.items()
        })
        
        # 因果发现组件
        self.causal_extractor = CausalVariableExtractor(
            input_dim=causal_dim,
            num_vars=num_causal_vars,
            prior_graph=self.prior_graph
        )
        
        self.structure_learner = EnhancedStructureLearner(
            num_vars=num_causal_vars,
            hidden_dim=causal_dim,
            prior_graph=self.prior_graph,
            prior_weight=prior_weight
        )

        self.intervention_calculator = InterventionEffectCalculator(
            num_vars=num_causal_vars,
            causal_dim=causal_dim
        )
        
        # CLEAR项目兼容的因果机制预测器
        self.mechanism_predictor = ClearMechanismPredictor(
            num_vars=num_causal_vars,
            hidden_dim=causal_dim
        )
        
        # 检索增强模块
        self.retrieval_enhancer = RetrievalEnhancer(
            causal_dim=causal_dim,
            num_vars=num_causal_vars
        )

    def _build_clear_encoder(self, input_dim: int) -> nn.Module:
        """构建与CLEAR项目兼容的编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, self.causal_dim * 2),
            nn.LayerNorm(self.causal_dim * 2),
            nn.ReLU(),
            nn.Linear(self.causal_dim * 2, self.causal_dim)
        )

    def forward(
        self,
        modal_features: Dict[str, torch.Tensor],
        retrieval_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            modal_features: 不同模态的输入特征
            retrieval_mode: 是否用于检索任务
        """
        # 1. 编码特征
        encoded_features = {
            modality: self.modal_encoders[modality](features)
            for modality, features in modal_features.items()
        }
        
        # 2. 提取因果变量
        causal_vars = {
            modality: self.causal_extractor(features)
            for modality, features in encoded_features.items()
        }
        
        # 3. 学习因果结构（结合先验图）
        aligned_vars = self._align_modal_variables(causal_vars)
        adj_matrix = self.structure_learner(aligned_vars)
        
        # 4. 预测因果机制
        mechanisms = self.mechanism_predictor(aligned_vars, adj_matrix)

        intervention_effects = self.intervention_calculator(
            aligned_vars,
            adj_matrix,
            intervention_idx=0,  # 可以根据需要设置要干预的变量索引
            intervention_value=torch.zeros_like(aligned_vars[:, 0])  # 示例干预值
        )
        
        # 5. 检索增强
        if retrieval_mode:
            retrieval_output = self.retrieval_enhancer(
                aligned_vars,
                adj_matrix,
                mechanisms
            )
            return {
                'causal_vars': aligned_vars,
                'adj_matrix': adj_matrix,
                'mechanisms': mechanisms,
                'intervention_effects': intervention_effects,
                'retrieval_scores': retrieval_output['scores'],
                'causal_attention': retrieval_output['attention']
            }
        
        return {
            'causal_vars': aligned_vars,
            'adj_matrix': adj_matrix,
            'mechanisms': mechanisms,
            'intervention_effects': intervention_effects
        }

    def _align_modal_variables(self, modal_vars: Dict[str, torch.Tensor]) -> torch.Tensor:
        """对齐不同模态的因果变量"""
        # 将所有模态的变量堆叠
        stacked_vars = torch.stack(list(modal_vars.values()), dim=1)
        
        # 通过注意力机制对齐
        B, M, N, D = stacked_vars.shape
        stacked_vars = stacked_vars.view(B * N, M, D)
        
        # 计算跨模态注意力
        attention = torch.matmul(stacked_vars, stacked_vars.transpose(-2, -1))
        attention = torch.softmax(attention / np.sqrt(D), dim=-1)
        
        # 对齐变量
        aligned = torch.matmul(attention, stacked_vars)
        aligned = aligned.mean(dim=1).view(B, N, D)
        
        return aligned

class EnhancedStructureLearner(nn.Module):
    """增强版结构学习器，整合预定义因果图"""
    
    def __init__(self, num_vars: int, hidden_dim: int, prior_graph: nx.DiGraph, prior_weight: float):
        super().__init__()
        self.prior_graph = prior_graph
        self.prior_weight = prior_weight
        
        # 转换预定义图为邻接矩阵
        self.register_buffer(
            'prior_adj',
            self._graph_to_adj(prior_graph, num_vars)
        )
        
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _graph_to_adj(self, graph: nx.DiGraph, size: int) -> torch.Tensor:
        """将NetworkX图转换为邻接矩阵"""
        adj = torch.zeros(size, size)
        for i, j in graph.edges():
            adj[i, j] = 1
        return adj

    def forward(self, causal_vars: torch.Tensor) -> torch.Tensor:
        # 计算数据驱动的邻接矩阵
        learned_adj = self._compute_learned_adj(causal_vars)
        
        # 结合先验知识
        final_adj = (1 - self.prior_weight) * learned_adj + \
                   self.prior_weight * self.prior_adj
                   
        return final_adj

class RetrievalEnhancer(nn.Module):
    """检索增强模块，用于检索任务"""
    
    def __init__(self, causal_dim: int, num_vars: int):
        super().__init__()
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=causal_dim,
            num_heads=8
        )
        
        self.score_projection =nn.Sequential(
            nn.Linear(causal_dim, causal_dim // 2),
            nn.ReLU(),
            nn.Linear(causal_dim // 2, 1)
        )
        
    def forward(
        self,
        causal_vars: torch.Tensor,
        adj_matrix: torch.Tensor,
        mechanisms: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        使用因果信息增强检索
        """
        # 应用因果注意力
        attn_output, attention_weights = self.causal_attention(
            causal_vars, causal_vars, causal_vars
        )
        
        # 结合因果图信息
        causal_context = attn_output * adj_matrix.unsqueeze(-1)
        
        # 计算检索分数
        retrieval_scores = self.score_projection(causal_context).squeeze(-1)
        
        return {
            'scores': retrieval_scores,
            'attention': attention_weights
        }

class InterventionEffectCalculator(nn.Module):
    def __init__(self, num_vars: int, causal_dim: int, max_propagation_steps: int = 3):
        super().__init__()
        self.num_vars = num_vars
        self.causal_dim = causal_dim
        self.max_steps = max_propagation_steps
        self.propagation_net = nn.Sequential(
            nn.Linear(causal_dim * 2, causal_dim),
            nn.ReLU(),
            nn.Linear(causal_dim, causal_dim)
        )
    
    def forward(self, causal_vars: torch.Tensor, adj_matrix: torch.Tensor, 
                intervention_idx: int, intervention_value: torch.Tensor) -> torch.Tensor:
        """计算干预效应"""
        intervened_vars = causal_vars.clone()
        intervened_vars[:, intervention_idx] = intervention_value
        
        # 迭代传播干预效果
        for _ in range(self.max_steps):
            # 获取因果关系影响
            causal_effects = torch.bmm(adj_matrix.unsqueeze(0), intervened_vars)
            # 更新受影响的变量
            intervened_vars = self.propagation_net(
                torch.cat([intervened_vars, causal_effects], dim=-1)
            )
            
        return intervened_vars