from typing import Dict, Any
import networkx as nx
import numpy as np
import torch
from avatar.models.definition import Definition

class CausalMediationAnalyzer:
    """
    因果中介分析类
    实现基于图结构的中介效应计算，用于优化模型性能
    """
    
    def __init__(self, definition: Definition):
        """
        初始化函数
        :param definition: Definition实例，包含因果关系图等信息
        """
        self.definition = definition
        self.graph = definition.graph
        self.mediation_effects = None
        
    def calculate_mediation_effects(self) -> Dict[str, float]:
        """
        计算中介效应
        :return: 包含中介效应值的字典
        """
        if not isinstance(self.graph, nx.DiGraph):
            raise ValueError("Input graph must be a networkx.DiGraph")
            
        # 计算直接效应
        direct_effect = self._calculate_direct_effect()
        
        # 计算间接效应
        indirect_effect = self._calculate_indirect_effect()
        
        # 计算总效应
        total_effect = direct_effect + indirect_effect
        
        self.mediation_effects = {
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'total_effect': total_effect,
            'effect_ratio': indirect_effect / total_effect if total_effect != 0 else 0
        }
        
        return self.mediation_effects
        
    def _calculate_direct_effect(self) -> float:
        """
        计算直接效应
        """
        if 'image' not in self.graph or 'semantic' not in self.graph:
            return 0.0
            
        try:
            # 获取直接路径的权重
            direct_paths = list(nx.all_simple_paths(
                self.graph, 
                source='image', 
                target='semantic',
                cutoff=1  # 只考虑直接路径
            ))
            
            if not direct_paths:
                return 0.0
                
            # 取第一条直接路径的权重
            path = direct_paths[0]
            weights = []
            for u, v in zip(path[:-1], path[1:]):
                weights.append(self.graph[u][v].get('weight', 0.0))
                
            return np.prod(weights)
            
        except nx.NetworkXNoPath:
            return 0.0
            
    def _calculate_indirect_effect(self) -> float:
        """
        计算间接效应
        """
        if 'image' not in self.graph or 'semantic' not in self.graph:
            return 0.0
            
        try:
            # 获取所有间接路径
            indirect_paths = list(nx.all_simple_paths(
                self.graph,
                source='image',
                target='semantic',
                cutoff=len(self.graph)  # 考虑所有可能路径
            ))
            
            if not indirect_paths:
                return 0.0
                
            total_effect = 0.0
            for path in indirect_paths:
                if len(path) <= 2:  # 跳过直接路径
                    continue
                    
                weights = []
                for u, v in zip(path[:-1], path[1:]):
                    weights.append(self.graph[u][v].get('weight', 0.0))
                    
                path_effect = np.prod(weights)
                total_effect += path_effect
                
            return total_effect
            
        except nx.NetworkXNoPath:
            return 0.0

    def optimize_model_weights(self) -> None:
        """
        根据中介效应分析结果优化模型权重
        """
        if not self.mediation_effects:
            self.calculate_mediation_effects()
            
        effect_ratio = self.mediation_effects['effect_ratio']
        
        # 调整边权重预测网络的参数
        for param in self.definition.edge_weight_net.parameters():
            # 根据中介效应比例调整学习率
            param.requires_grad = True
            if effect_ratio > 0.5:  # 如果间接效应占主导
                param.data *= (1 + effect_ratio * 0.1)  # 增强间接路径权重
            else:
                param.data *= (1 - effect_ratio * 0.1)  # 增强直接路径权重

def causal_mediation_analysis(definition: Definition) -> Dict[str, float]:
    """
    因果中介分析函数
    :param definition: Definition实例
    :return: 包含中介效应值的字典
    """
    analyzer = CausalMediationAnalyzer(definition)
    return analyzer.calculate_mediation_effects()
