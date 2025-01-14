import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from transformers import CLIPModel, CLIPTokenizer
import numpy as np
from typing import Dict, Tuple
from cmscm import CausalVariable, CMSCM

class Definition(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256, alignment_temp=0.07):
        super().__init__()
        # 保持原有基础组件
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
       
        self.modality_disentangler = nn.ModuleDict({
            'image': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'text': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })
        
        
        self.multi_head_attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=8)
        self.modality_suppression = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 语义投影和场景分解
        self.semantic_projection = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        self.scene_decomposer = HierarchicalSceneDecomposer(shared_dim, hidden_dim)
        
        # 集成CM-SCM的变量处理模块
        self.X = CausalVariable(hidden_dim)  # 图像模态
        self.Y = CausalVariable(hidden_dim)  # 文本模态
        self.S = CausalVariable(shared_dim)  # 共享语义
        
        self.cmscm = CMSCM(hidden_dim, shared_dim)
        
        # 保持原有图结构
        self.graph = nx.DiGraph()
        self._initialize_graph()
        
        # 边权重预测网络
        self.edge_weight_net = nn.Sequential(
            nn.Linear(shared_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.alignment_temp = alignment_temp
        self.modality_discriminator = ModalityDiscriminator(hidden_dim)

    def _initialize_graph(self):
        """保持原有的图初始化"""
        nodes = ['image', 'text', 'semantic', 'scene']
        self.graph.add_nodes_from(nodes)
        base_edges = [
            ('image', 'semantic'),
            ('text', 'semantic'),
            ('semantic', 'scene')
        ]
        self.graph.add_edges_from(base_edges)
        
    def process_features(self, x, modality):
        """结合原有特征处理和CM-SCM噪声处理"""
        # 原有的模态解耦
        x_disentangled = self.disentangle_modality(x, modality)
        x_suppressed = self.suppress_modality(x_disentangled)
        
        # CM-SCM噪声处理
        if modality == 'image':
            x_noised = self.X.add_noise(x_suppressed)
        else:
            x_noised = self.Y.add_noise(x_suppressed)
            
        return x_noised, x_suppressed

    def compute_structural_equations(self, img_feat, txt_feat):
        return self.cmscm.structural_equations(img_feat, txt_feat)

    def encode_image(self, image):
        """增强的图像编码"""
        img_feat = self.clip.get_image_features(image)
        img_noised, img_suppressed = self.process_features(img_feat, 'image')
        img_semantic = self.semantic_projection(img_suppressed)
        img_scene = self.scene_decomposer(img_semantic)
        return img_semantic, img_scene, img_noised

    def encode_text(self, text):
        """增强的文本编码"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        txt_feat = self.clip.get_text_features(**inputs)
        txt_noised, txt_suppressed = self.process_features(txt_feat, 'text')
        txt_semantic = self.semantic_projection(txt_suppressed)
        txt_scene = self.scene_decomposer(txt_semantic)
        return txt_semantic, txt_scene, txt_noised

    def compute_mediation_effects(self, features):
        """整合CausalMediationAnalyzer和CM-SCM的中介效应计算"""
        effects = {}
        
        # 变量映射
        X = features['img_noised']  # 图像模态
        Y = features['txt_noised']  # 文本模态
        S = features['semantic']    # 共享语义
        Zx = features['img_semantic']  # 图像特定语义
        Zy = features['txt_semantic']  # 文本特定语义
        
        # 1. 图路径分析 (CausalMediationAnalyzer)
        for edge in self.graph.edges():
            source, target = edge
            if source in features and target in features:
                combined = torch.cat([features[source], features[target]], dim=-1)
                weight = self.edge_weight_net(combined)
                self.graph[source][target]['weight'] = weight.item()
                
        # 2. 结构方程计算 (CM-SCM)
        struct_outputs = self.compute_structural_equations(X, Y)
        
        # 3. 双重验证机制
        # 3.1 图路径效应
        graph_effects = {
            'direct': {
                'img_to_semantic': self.graph['image']['semantic']['weight'],
                'txt_to_semantic': self.graph['text']['semantic']['weight']
            },
            'indirect': {
                'img_to_scene': self.graph['image']['semantic']['weight'] * 
                               self.graph['semantic']['scene']['weight'],
                'txt_to_scene': self.graph['text']['semantic']['weight'] * 
                               self.graph['semantic']['scene']['weight']
            }
        }
        
        # 3.2 结构方程效应
        structural_effects = {
            'S': struct_outputs['S'],
            'Zx': struct_outputs['Zx'],
            'Zy': struct_outputs['Zy'],
            'R': struct_outputs['R']
        }
        
        # 4. 效应一致性验证
        consistency_loss = F.mse_loss(
            torch.tensor([
                graph_effects['direct']['img_to_semantic'],
                graph_effects['direct']['txt_to_semantic']
            ]),
            torch.tensor([
                structural_effects['S'].mean().item(),
                structural_effects['R'].mean().item()
            ])
        )
        
        # 5. 返回整合结果
        effects.update({
            'graph_effects': graph_effects,
            'structural_effects': structural_effects,
            'consistency_loss': consistency_loss.item()
        })
        
        return effects

    def forward(self, images, texts):
        """增强的前向传播"""
        # 1. 特征编码
        img_semantic, img_scene, img_noised = self.encode_image(images)
        txt_semantic, txt_scene, txt_noised = self.encode_text(texts)

        # 2. 对抗训练
        modality_labels = torch.tensor([0] * len(img_semantic) + [1] * len(txt_semantic)).to(img_semantic.device)
        all_features = torch.cat([img_semantic, txt_semantic], dim=0)
        adv_loss = self.adversarial_loss(all_features, modality_labels)

        # 3. 构建特征字典
        features = {
            'image': img_semantic,
            'text': txt_semantic,
            'semantic': (img_semantic + txt_semantic) / 2,
            'scene': (img_scene + txt_scene) / 2,
            'img_semantic': img_semantic,
            'txt_semantic': txt_semantic,
            'img_scene': img_scene,
            'txt_scene': txt_scene,
            'img_noised': img_noised,
            'txt_noised': txt_noised
        }

        # 4. 计算中介效应
        mediation_effects = self.compute_mediation_effects(features)

        # 5. 返回结果
        return {
            **features,
            'effects': mediation_effects,
            'graph': self.graph,
            'adv_loss': adv_loss
        }

def definition():
    return Definition()
