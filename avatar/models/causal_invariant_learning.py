from .causal_graph_learning import CausalGraphLearning
from .causal_mechanism import CausalMechanism
from .retrieval_model import RetrievalModel
from .attention import CausalAttention
import torch
import torch.nn as nn

class CausalInvariantLearning(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 使用AVATAR的配置
        self.hidden_size = config.hidden_size
        
        # 底层：模态内部因果关系建模
        self.intra_modal_causal = {
            'image': IntraModalCausal(config.hidden_size),
            'text': IntraModalCausal(config.hidden_size)
        }
        
        # 使用AVATAR的attention
        self.attention = AvatarAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads
        )
        
        # 高层：跨模态统一
        self.cross_modal_causal = CrossModalCausal(
            modal_dims={
                'image': config.hidden_size,
                'text': config.hidden_size
            },
            unified_dim=config.hidden_size,
            avatar_attention=self.attention  # 传入AVATAR的attention
        )
        
        # 不变性约束
        self.invariance_regularizer = InvarianceRegularizer(config.hidden_size)
        
        # AVATAR的投影层
        self.image_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.text_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, features):
        # 1. 特征投影
        projected_features = {
            'image': self.image_projection(features['image']),
            'text': self.text_projection(features['text'])
        }
        
        # 2. 底层模态内因果学习
        intra_causal_features = {}
        for modal, feat in projected_features.items():
            intra_causal_features[modal] = self.intra_modal_causal[modal](feat)
        
        # 3. 高层跨模态统一
        unified_causal_repr = self.cross_modal_causal(intra_causal_features)
        
        # 4. 计算不变性损失
        invariance_loss = self.invariance_regularizer(
            unified_causal_repr,
            intra_causal_features
        )
        
        return {
            'unified_repr': unified_causal_repr,
            'intra_causal_features': intra_causal_features,
            'invariance_loss': invariance_loss
        }

class IntraModalCausal(nn.Module):
    """底层：单模态内部因果关系建模"""
    def __init__(self, feature_dim):
        super().__init__()
        self.causal_graph = CausalGraphLearning(feature_dim)
        self.causal_mechanism = CausalMechanism(feature_dim)
    
    def forward(self, features):
        causal_graph = self.causal_graph(features)
        causal_repr = self.causal_mechanism(features, causal_graph)
        return causal_repr

class CrossModalCausal(nn.Module):
    def __init__(self, modal_dims, unified_dim, avatar_attention):
        super().__init__()
        # 直接使用AVATAR的projection层
        self.modal_projectors = nn.ModuleDict({
            modal: nn.Linear(dim, unified_dim) 
            for modal, dim in modal_dims.items()
        })
        # 使用AVATAR的attention
        self.attention = avatar_attention
        
    def forward(self, modal_features):
        # 投影到统一空间
        unified_features = {
            modal: self.modal_projectors[modal](feat)
            for modal, feat in modal_features.items()
        }
        # 使用AVATAR的attention机制
        return self.attention(unified_features)

class InvarianceRegularizer(nn.Module):
    """理论保证：数学上的不变性约束"""
    def __init__(self, feature_dim):
        super().__init__()
        
    def forward(self, unified_repr, modal_features):
        # 计算跨模态因果不变性损失
        invariance_loss = 0
        for modal_feat in modal_features.values():
            invariance_loss += self.compute_invariance_loss(
                unified_repr, modal_feat
            )
        return invariance_loss
    
    def compute_invariance_loss(self, unified_repr, modal_feat):
        # 实现数学上的不变性度量
        # 可以使用MMD或其他统计距离
        return mmd_distance(unified_repr, modal_feat)