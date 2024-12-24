import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image
import spacy
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from causal_graph_definition import Definition
from causal_discovery import Discovery

class RetrievalModel(nn.Module):
    def __init__(self):
        super(RetrievalModel, self).__init__()
        self.definition = Definition()  # 包含了语义编码、场景分解、模态抑制
        self.discovery = Discovery()    # 包含因果图学习
        
        # 定义用于计算因果强度的线性层
        self.causal_strength = nn.Linear(256 * 2, 1)  # 假设每个特征向量是256维

    def forward(self, images, texts):
        """
        Forward pass for computing similarity and causal scores.
        
        Args:
            images (torch.Tensor): Batch of image features.
            texts (torch.Tensor): Batch of text features.

        Returns:
            dict: A dictionary containing the final scores, similarity scores, and causal scores.
        """
        # 特征编码
        img_features = self.definition.encode_image(images)
        text_features = self.definition.encode_text(texts)
        
        # 场景分解和模态抑制
        img_features = self.definition.suppress_modality(img_features)
        text_features = self.definition.suppress_modality(text_features)

        # 计算相似度分数（余弦相似度）
        similarity_scores = torch.matmul(img_features, text_features.T)
        norm_img = torch.norm(img_features, dim=1, keepdim=True)
        norm_text = torch.norm(text_features, dim=1, keepdim=True)
        similarity_scores = similarity_scores / (norm_img @ norm_text.T)

        # 获取因果图信息和边权重
        discovery_output = self.discovery(images, texts)
        edge_weights = discovery_output.get('edge_weights', {})  # 确保有默认值

        # 计算因果分数
        causal_scores = self.compute_causal_scores(img_features, text_features, edge_weights)

        # 对齐约束下的综合分数
        final_scores = similarity_scores + 0.3 * causal_scores
        
        return {
            'scores': final_scores,
            'similarity': similarity_scores,
            'causal': causal_scores
        }

    def compute_causal_scores(self, img_f, text_f, edge_weights):
        """
        Compute causal scores based on the edge weights and joint features.
        
        Args:
            img_f (torch.Tensor): Image feature tensor.
            text_f (torch.Tensor): Text feature tensor.
            edge_weights (dict): Dictionary of edge weights from the discovery model.

        Returns:
            torch.Tensor: Causal scores.
        """
        batch_size = img_f.size(0)
        causal_strengths = torch.zeros((batch_size, batch_size), device=img_f.device)
        for i in range(batch_size):
            for j in range(batch_size):
                weight = edge_weights.get((i, j), 0.0)  # 默认值为0.0如果没有对应的边
                combined = torch.cat([img_f[i], text_f[j]], dim=-1).unsqueeze(0)
                strength = self.causal_strength(combined).squeeze(-1)
                causal_strengths[i, j] = weight * strength

        return causal_strengths

    def compute_loss(self, outputs, labels):
        """
        Compute multi-objective loss function.
        
        Args:
            outputs (dict): Output dictionary from the forward pass.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Total loss.
        """
        # 注意，这里假设labels是一个表示正样本位置的长整型张量。
        # 如果不是这种情况，你需要根据实际情况调整这一部分。
        sim_loss = F.cross_entropy(outputs['similarity'], labels)
        causal_loss = F.cross_entropy(outputs['causal'], labels)
        semantic_loss = self.definition.compute_alignment_loss() if hasattr(self.definition, 'compute_alignment_loss') else 0
        
        total_loss = sim_loss + 0.3 * causal_loss + 0.1 * semantic_loss
        return total_loss