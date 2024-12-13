import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from transformers import CLIPModel

class UnifiedCausalGraph(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared_dim = shared_dim
        
        # 基础编码器
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # 语义编码层
        self.semantic_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        # 模态抑制
        self.modality_suppression = nn.MultiheadAttention(hidden_dim, 8)
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 场景分解
        self.scene_decomposer = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        # 边权重更新网络
        self.edge_weight_net = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化因果图
        self.graph = nx.DiGraph()
        self._initialize_graph()
        
    def _initialize_graph(self):
        """初始化基础图结构"""
        nodes = ['image', 'text', 'semantic', 'scene']
        self.graph.add_nodes_from(nodes)
        base_edges = [
            ('image', 'semantic'),
            ('text', 'semantic'),
            ('semantic', 'scene')
        ]
        self.graph.add_edges_from(base_edges)
        
    def suppress_modality(self, x):
        attn_mask = self.modality_gate(x)
        attended, _ = self.modality_suppression(x, x, x)
        return x - (attended * attn_mask)
        
    def encode_features(self, images, texts):
        # 获取并抑制模态特征
        img_feat = self.suppress_modality(self.clip.get_image_features(images))
        txt_feat = self.suppress_modality(self.clip.get_text_features(texts))
        
        # 映射到共享空间
        img_semantic = self.semantic_projection(img_feat)
        txt_semantic = self.semantic_projection(txt_feat)
        
        # 场景分解
        img_scene = self.scene_decomposer(img_semantic)
        txt_scene = self.scene_decomposer(txt_semantic)
        
        return {
            'image': img_feat,
            'text': txt_feat,
            'img_semantic': img_semantic,
            'txt_semantic': txt_semantic,
            'scene': (img_scene + txt_scene) / 2
        }
    
    def update_edge_weights(self, features):
        """动态更新边权重"""
        for edge in self.graph.edges():
            source, target = edge
            if source in features and target in features:
                combined = torch.cat([features[source], features[target]], dim=-1)
                weight = self.edge_weight_net(combined)
                self.graph[source][target]['weight'] = weight.item()
    
    def forward(self, images, texts):
        # 1. 编码并获取模态无关特征
        features = self.encode_features(images, texts)
        
        # 2. 更新因果图边权重
        self.update_edge_weights(features)
        
        # 3. 计算对齐损失
        semantic_loss = F.mse_loss(features['img_semantic'], features['txt_semantic'])
        
        return {
            'features': features,
            'graph': self.graph,
            'semantic_loss': semantic_loss
        }

def build_model():
    return UnifiedCausalGraph()