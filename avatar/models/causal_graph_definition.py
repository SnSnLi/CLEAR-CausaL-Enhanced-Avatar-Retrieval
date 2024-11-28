# causal_graph_definition.py

import networkx as nx
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class CausalVariable:
    """定义因果图中的变量节点"""
    name: str
    type: str  # 'image', 'text', 'spatial', 'action', or 'latent'
    dimension: int
    is_observable: bool = True
    description: str = ""  
    """比如吧就"image_features": "图像的视觉编码表示"
"text_features": "文本的语义编码表示"
"spatial_features": "实体的位置关系特征"
"action_features": "人物/物体的动作状态特征"
"semantic_content": "图文共享的高层语义信息""""
    
class CausalRelation:
    """定义因果关系"""
    def __init__(self, cause: str, effect: str, strength: float = 1.0, description: str = ""):
        self.cause = cause
        self.effect = effect
        self.strength = strength
        self.description = description
        self.mechanism: Optional[nn.Module] = None

class CausalGraph:
    """因果图的核心定义类"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.variables: Dict[str, CausalVariable] = {}
        self.relations: List[CausalRelation] = []
        
        # 初始化多模态检索的基本变量
        self._initialize_basic_variables()
        
    def _initialize_basic_variables(self):
        
        # 图像基础特征
        self.add_variable(CausalVariable(
            name="image_features",
            type="image",
            dimension=512,
            is_observable=True,
            description="原始图像的视觉特征表示"
        ))
        
        # 文本基础特征
        self.add_variable(CausalVariable(
            name="text_features",
            type="text",
            dimension=512,
            is_observable=True,
            description="文本描述的语义特征表示"
        ))
        
        # 空间关系特征
        self.add_variable(CausalVariable(
            name="spatial_features",
            type="spatial",
            dimension=256,
            is_observable=True,
            description="实体间的空间位置关系特征"
        ))
        
        # 实体位置特征
        self.add_variable(CausalVariable(
            name="entity_location",
            type="spatial",
            dimension=128,
            is_observable=True,
            description="图像中具体实体的位置信息"
        ))
        
        # 动作特征
        self.add_variable(CausalVariable(
            name="action_features",
            type="action",
            dimension=256,
            is_observable=True,
            description="实体的动作行为特征"
        ))
        
        # 实体交互特征
        self.add_variable(CausalVariable(
            name="interaction_features",
            type="action",
            dimension=128,
            is_observable=True,
            description="实体间的交互行为特征"
        ))
        
        # 语义内容特征
        self.add_variable(CausalVariable(
            name="semantic_content",
            type="latent",
            dimension=512,
            is_observable=False,
            description="图文共同的高层语义表示"
        ))
        
        # 场景上下文
        self.add_variable(CausalVariable(
            name="context",
            type="latent",
            dimension=256,
            is_observable=False,
            description="场景的整体上下文信息"
        ))

    def add_variable(self, variable: CausalVariable):
        """添加变量到因果图"""
        self.variables[variable.name] = variable
        self.graph.add_node(variable.name)
        
    def add_relation(self, relation: CausalRelation):
        """添加因果关系"""
        if relation.cause not in self.variables or relation.effect not in self.variables:
            raise ValueError("Cause or effect variable not found in graph")
        
        self.relations.append(relation)
        self.graph.add_edge(relation.cause, relation.effect, 
                           weight=relation.strength,
                           description=relation.description)
        
    def get_ancestors(self, variable: str) -> List[str]:
        """获取给定变量的所有祖先节点"""
        return list(nx.ancestors(self.graph, variable))
    
    def get_descendants(self, variable: str) -> List[str]:
        """获取给定变量的所有后代节点"""
        return list(nx.descendants(self.graph, variable))
    
    def get_causal_paths(self, source: str, target: str) -> List[List[str]]:
        """获取从源节点到目标节点的所有因果路径"""
        return list(nx.all_simple_paths(self.graph, source, target))
    
    def get_intervention_effects(self, intervention_var: str) -> Dict[str, float]:
        """计算干预某个变量对其他变量的影响程度"""
        effects = {}
        descendants = self.get_descendants(intervention_var)
        
        for desc in descendants:
            paths = self.get_causal_paths(intervention_var, desc)
            total_effect = 0
            for path in paths:
                path_effect = 1.0
                for i in range(len(path)-1):
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    path_effect *= edge_data['weight']
                total_effect += path_effect
            effects[desc] = total_effect
            
        return effects
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """将因果图转换为邻接矩阵"""
        return nx.adjacency_matrix(self.graph).todense()
    
    def validate_graph(self) -> bool:
        """验证因果图的有效性"""
        # 检查是否是有向无环图(DAG)
        if not nx.is_directed_acyclic_graph(self.graph):
            return False
        
        # 检查必要的变量是否存在
        required_vars = {
            "image_features", 
            "text_features", 
            "spatial_features",
            "action_features"
        }
        if not required_vars.issubset(self.variables.keys()):
            return False
            
        return True

def create_multimodal_retrieval_graph() -> CausalGraph:
    """创建专门针对Flickr30k_entities的多模态检索因果图实例"""
    graph = CausalGraph()
    
    # 添加基本因果关系
    relations = [
        # 图像特征相关的因果关系
        CausalRelation(
            "image_features", 
            "semantic_content", 
            0.8,
            "图像特征对语义内容的直接影响"
        ),
        CausalRelation(
            "image_features",
            "spatial_features",
            0.7,
            "图像特征决定空间关系特征"
        ),
        CausalRelation(
            "image_features",
            "action_features",
            0.7,
            "图像特征决定动作特征"
        ),
        
        # 文本特征相关的因果关系
        CausalRelation(
            "text_features",
            "semantic_content",
            0.8,
            "文本特征影响语义内容的直接关系"
        ),
        CausalRelation(
            "text_features",
            "spatial_features",
            0.6,
            "文本描述中的空间关系信息"
        ),
        CausalRelation(
            "text_features",
            "action_features",
            0.6,
            "文本描述中的动作信息"
        ),
        
        # 空间特征相关的因果关系
        CausalRelation(
            "spatial_features",
            "entity_location",
            0.8,
            "空间特征决定实体位置信息"
        ),
        CausalRelation(
            "spatial_features",
            "semantic_content",
            0.6,
            "空间关系对语义理解的影响"
        ),
        
        # 动作特征相关的因果关系
        CausalRelation(
            "action_features",
            "interaction_features",
            0.8,
            "动作特征决定实体间交互特征"
        ),
        CausalRelation(
            "action_features",
            "semantic_content",
            0.6,
            "动作信息对语义理解的影响"
        ),
        
        # 上下文相关的因果关系
        CausalRelation(
            "context",
            "image_features",
            0.5,
            "上下文对图像特征的影响"
        ),
        CausalRelation(
            "context",
            "text_features",
            0.5,
            "上下文对文本特征的影响"
        ),
    ]
    
    for relation in relations:
        graph.add_relation(relation)
        
    return graph

# 用于其他模块调用的接口函数
def get_causal_graph() -> CausalGraph:
    """获取因果图实例的全局接口"""
    return create_multimodal_retrieval_graph()