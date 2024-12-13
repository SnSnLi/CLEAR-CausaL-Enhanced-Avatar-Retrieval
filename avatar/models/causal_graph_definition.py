import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class ModalityInvariantEncoder(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # 语义编码层
        self.semantic_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        # 模态抑制注意力
        self.modality_suppression = nn.MultiheadAttention(hidden_dim, 8)
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 场景分解模块
        self.scene_decomposer = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim)
        )
        
        # 对齐损失
        self.alignment_temp = 0.07
        
    def suppress_modality(self, x):
        # 模态特异性抑制
        attn_mask = self.modality_gate(x)
        attended, _ = self.modality_suppression(x, x, x)
        return x - (attended * attn_mask)
    
    def encode_image(self, image):
        img_feat = self.clip.get_image_features(image)
        # 抑制模态特征
        img_feat = self.suppress_modality(img_feat)
        # 映射到共享空间
        img_semantic = self.semantic_projection(img_feat)
        # 场景分解
        img_scene = self.scene_decomposer(img_semantic)
        return img_semantic, img_scene
        
    def encode_text(self, text):
        txt_feat = self.clip.get_text_features(text)
        txt_feat = self.suppress_modality(txt_feat)
        txt_semantic = self.semantic_projection(txt_feat)
        txt_scene = self.scene_decomposer(txt_semantic)
        return txt_semantic, txt_scene
    
    def compute_alignment_loss(self, img_feat, txt_feat):
        # 计算对齐损失
        similarity = torch.matmul(img_feat, txt_feat.T) / self.alignment_temp
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        loss = F.cross_entropy(similarity, labels)
        return loss
    
    def forward(self, images, texts):
        img_semantic, img_scene = self.encode_image(images)
        txt_semantic, txt_scene = self.encode_text(texts)
        
        # 语义对齐损失
        semantic_loss = self.compute_alignment_loss(img_semantic, txt_semantic)
        # 场景对齐损失
        scene_loss = self.compute_alignment_loss(img_scene, txt_scene)
        
        return {
            'img_semantic': img_semantic,
            'txt_semantic': txt_semantic,
            'img_scene': img_scene,
            'txt_scene': txt_scene,
            'semantic_loss': semantic_loss,
            'scene_loss': scene_loss
        }

def build_model():
    return ModalityInvariantEncoder()