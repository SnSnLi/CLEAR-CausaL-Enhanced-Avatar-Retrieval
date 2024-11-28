import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

class CounterfactualReasoning(nn.Module):
    def __init__(self, causal_graph, feature_dim: int):
        super().__init__()
        self.causal_graph = causal_graph
        self.feature_dim = feature_dim
        
        # 反事实特征生成器 - 为每个主要变量创建独立生成器
        self.generators = nn.ModuleDict({
            'image': self._build_generator(),
            'text': self._build_generator(),
            'spatial': self._build_generator(),
            'action': self._build_generator()
        })
        
        # 特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 因果效应评估器
        self.effect_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def _build_generator(self):
        return nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def generate_counterfactual(self, features: Dict[str, torch.Tensor], 
                              intervention_var: str) -> Dict[str, torch.Tensor]:
        """生成反事实特征"""
        cf_features = {}
        
        # 获取干预变量的影响范围
        affected_vars = self.causal_graph.get_descendants(intervention_var)
        
        # 为受影响的变量生成反事实特征
        for var in affected_vars:
            if var in self.generators:
                # 结合原始特征和干预信息生成反事实特征
                orig_feature = features[var]
                intervention_feature = features[intervention_var]
                
                combined = torch.cat([orig_feature, intervention_feature], dim=-1)
                cf_features[var] = self.generators[var](combined)
        
        return cf_features

    def compute_counterfactual_effect(self, 
                                    orig_features: Dict[str, torch.Tensor],
                                    cf_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算反事实效应"""
        # 融合原始特征
        orig_fused = self.feature_fusion(torch.cat([
            orig_features['image'],
            orig_features['text']
        ], dim=-1))
        
        # 融合反事实特征
        cf_fused = self.feature_fusion(torch.cat([
            cf_features['image'],
            cf_features['text']
        ], dim=-1))
        
        # 评估效应差异
        effect = self.effect_estimator(torch.cat([orig_fused, cf_fused], dim=-1))
        return effect

    def forward(self, features: Dict[str, torch.Tensor], 
               intervention_var: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        前向传播
        Args:
            features: 原始特征字典
            intervention_var: 要干预的变量名
        Returns:
            cf_features: 反事实特征
            effect: 反事实效应
        """
        # 生成反事实特征
        cf_features = self.generate_counterfactual(features, intervention_var)
        
        # 计算反事实效应
        effect = self.compute_counterfactual_effect(features, cf_features)
        
        return cf_features, effect

    def counterfactual_loss(self, 
                           orig_features: Dict[str, torch.Tensor],
                           cf_features: Dict[str, torch.Tensor],
                           labels: torch.Tensor,
                           intervention_var: str) -> torch.Tensor:
        """
        创新的反事实损失函数
        
        包含三个部分：
        1. 特征一致性损失 - 确保反事实特征在非干预维度上保持一致
        2. 因果效应损失 - 基于标签的反事实效应
        3. 干预敏感性损失 - 确保模型对关键干预敏感
        """
        # 1. 特征一致性损失
        consistency_loss = 0
        unaffected_vars = set(orig_features.keys()) - set(self.causal_graph.get_descendants(intervention_var))
        for var in unaffected_vars:
            if var in cf_features:
                consistency_loss += nn.functional.mse_loss(
                    orig_features[var], cf_features[var]
                )
        
        # 2. 因果效应损失
        effect = self.compute_counterfactual_effect(orig_features, cf_features)
        causal_loss = nn.functional.binary_cross_entropy(
            effect, labels
        )
        
        # 3. 干预敏感性损失 - 鼓励对关键干预产生显著变化
        intervention_strength = self.causal_graph.get_intervention_effects(intervention_var)
        sensitivity_loss = 0
        for var, strength in intervention_strength.items():
            if var in cf_features:
                target_diff = torch.abs(orig_features[var] - cf_features[var]).mean()
                sensitivity_loss += nn.functional.smooth_l1_loss(
                    target_diff, 
                    torch.tensor(strength).to(target_diff.device)
                )
        
        # 加权组合损失
        total_loss = (
            0.3 * consistency_loss + 
            0.5 * causal_loss + 
            0.2 * sensitivity_loss
        )
        
        return total_loss

    def explain_decision(self, 
                        orig_features: Dict[str, torch.Tensor],
                        cf_features: Dict[str, torch.Tensor],
                        intervention_var: str) -> Dict[str, float]:
        """
        解释模型决策的创新方法
        返回每个变量对最终决策的贡献度
        """
        explanations = {}
        
        # 获取干预的影响范围
        affected_vars = self.causal_graph.get_descendants(intervention_var)
        
        # 计算每个变量的贡献
        for var in affected_vars:
            if var in cf_features:
                # 计算特征变化程度
                feature_diff = torch.norm(
                    orig_features[var] - cf_features[var]
                ).item()
                
                # 获取因果图中的关系强度
                causal_strength = self.causal_graph.get_intervention_effects(
                    intervention_var
                ).get(var, 0.0)
                
                # 综合评估贡献度
                contribution = feature_diff * causal_strength
                explanations[var] = contribution
                
        # 归一化贡献度
        total = sum(explanations.values())
        if total > 0:
            explanations = {k: v/total for k, v in explanations.items()}
            
        return explanations