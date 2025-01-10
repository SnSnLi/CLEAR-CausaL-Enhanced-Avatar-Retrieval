import torch
import numpy as np
from typing import Dict, Any
from .analysis import CausalMediationAnalyzer
from .discovery import Discovery

class CausalScoreCalculator:
    """
    因果分数计算器
    整合所有相关权重计算，包括：
    - 直接效应权重
    - 间接效应权重
    - 图像质量权重
    - 短语相关性权重
    - 语义嵌入相似度
    - 图结构权重
    - 语义对齐损失
    """
    
    def __init__(self, discovery: Discovery):
        """
        初始化函数
        :param discovery: Discovery实例，包含所有相关权重信息
        """
        self.discovery = discovery
        self.analyzer = CausalMediationAnalyzer(discovery)
        
        # 初始化权重参数
        self.direct_effect_weight = 0.3
        self.indirect_effect_weight = 0.4
        self.image_quality_weight = 0.2
        self.phrase_relevance_weight = 0.3
        self.semantic_similarity_weight = 0.4
        self.graph_structure_weight = 0.3
        self.semantic_loss_weight = 0.2
        
    def calculate_causal_score(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        计算综合因果分数
        :param output: discovery的输出字典
        :return: 包含所有权重和最终得分的字典
        """
        # 获取mediation effects
        mediation_effects = self.analyzer.calculate_mediation_effects()
        
        # 计算语义嵌入相似度
        semantic_similarity = self._calculate_semantic_similarity(
            output['img_semantic'],
            output['txt_semantic']
        )
        
        # 计算图结构得分
        graph_structure_score = self._calculate_graph_structure_score(
            output['graph'],
            output['edge_weights']
        )
        
        # 计算各项得分
        direct_score = mediation_effects['direct_effect'] * self.direct_effect_weight
        indirect_score = mediation_effects['indirect_effect'] * self.indirect_effect_weight
        image_quality_score = output.get('image_quality', 1.0) * self.image_quality_weight
        phrase_relevance_score = output.get('phrase_relevance', 1.0) * self.phrase_relevance_weight
        semantic_similarity_score = semantic_similarity * self.semantic_similarity_weight
        graph_structure_score = graph_structure_score * self.graph_structure_weight
        semantic_loss_score = output['semantic_loss'] * self.semantic_loss_weight
        
        # 计算综合因果分数
        causal_score = (
            direct_score + 
            indirect_score + 
            image_quality_score + 
            phrase_relevance_score +
            semantic_similarity_score +
            graph_structure_score -
            semantic_loss_score  # 损失值需要减去
        )
        
        return {
            'causal_score': causal_score,
            'direct_effect': direct_score,
            'indirect_effect': indirect_score,
            'image_quality': image_quality_score,
            'phrase_relevance': phrase_relevance_score,
            'semantic_similarity': semantic_similarity_score,
            'graph_structure': graph_structure_score,
            'semantic_loss': semantic_loss_score,
            'mediation_effects': mediation_effects
        }

    def _calculate_semantic_similarity(self, img_semantic: torch.Tensor, txt_semantic: torch.Tensor) -> float:
        """
        计算语义嵌入相似度
        """
        return torch.cosine_similarity(img_semantic, txt_semantic, dim=-1).mean().item()

    def _calculate_graph_structure_score(self, graph: Any, edge_weights: list) -> float:
        """
        计算图结构得分
        """
        if not edge_weights:
            return 0.0
        return np.mean(edge_weights)

    def update_weights(self, **kwargs) -> None:
        """
        动态更新权重参数
        :param kwargs: 要更新的权重参数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid weight parameter: {key}")

def calculate_causal_score(discovery: Discovery, output: Dict[str, Any]) -> Dict[str, float]:
    """
    因果分数计算函数
    :param discovery: Discovery实例
    :param output: discovery的输出字典
    :return: 包含所有权重和最终得分的字典
    """
    calculator = CausalScoreCalculator(discovery)
    return calculator.calculate_causal_score(output)
