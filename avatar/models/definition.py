import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
import networkx as nx

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        attn_output, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        return attn_output.squeeze(0)

class HierarchicalSceneDecomposer(nn.Module):
    def __init__(self, shared_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.residual_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU()
        )
        self.residual_block2 = ResidualBlock(shared_dim, shared_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.residual_block1(x)
        x = self.layer2(x)
        x = self.residual_block2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        x += residual
        return x

class ModalityDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  
        )

    def forward(self, x):
        return self.classifier(x)

class Definition(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256, alignment_temp=0.07):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        # 模态解耦层
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

        # 多头注意力机制
        self.multi_head_attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=8)

        # 模态特异性抑制
        self.modality_suppression = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 语义编码层
        self.semantic_projection = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )

        # 场景分解模块
        self.scene_decomposer = HierarchicalSceneDecomposer(shared_dim, hidden_dim)

        # 初始化因果图
        self.graph = nx.DiGraph()
        self._initialize_graph()

        # 边权重预测网络
        self.edge_weight_net = nn.Sequential(
            nn.Linear(shared_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出一个介于0到1之间的权重值
        )

        # 对齐温度参数
        self.alignment_temp = alignment_temp

        # 对抗训练的分类器
        self.modality_discriminator = ModalityDiscriminator(hidden_dim)

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

    def disentangle_modality(self, x, modality):
        """模态解耦"""
        x = self.modality_disentangler[modality](x)
        # 应用多头注意力机制
        x = self.multi_head_attention(x)
        return x

    def suppress_modality(self, x):
        """模态特异性抑制"""
        attn_mask = self.modality_gate(x)
        attended, _ = self.modality_suppression(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attended = attended.squeeze(0)
        return x - (attended * attn_mask)

    def encode_image(self, image):
        img_feat = self.clip.get_image_features(image)
        # 解耦模态特征
        img_disentangled = self.disentangle_modality(img_feat, 'image')
        # 抑制模态特征
        img_suppressed = self.suppress_modality(img_disentangled)
        # 映射到共享空间
        img_semantic = self.semantic_projection(img_suppressed)
        # 场景分解
        img_scene = self.scene_decomposer(img_semantic)
        return img_semantic, img_scene

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        txt_feat = self.clip.get_text_features(**inputs)
        # 解耦模态特征
        txt_disentangled = self.disentangle_modality(txt_feat, 'text')
        # 抑制模态特征
        txt_suppressed = self.suppress_modality(txt_disentangled)
        # 映射到共享空间
        txt_semantic = self.semantic_projection(txt_suppressed)
        txt_scene = self.scene_decomposer(txt_semantic)
        return txt_semantic, txt_scene

    def update_edge_weights(self, features):
        """动态更新边权重"""
        for edge in self.graph.edges():
            source, target = edge
            if source in features and target in features:
                combined = torch.cat([features[source], features[target]], dim=-1)
                weight = self.edge_weight_net(combined)
                self.graph[source][target]['weight'] = weight.item()

    def compute_alignment_loss(self, img_feat, txt_feat):
        # 计算对齐损失
        similarity = torch.matmul(img_feat, txt_feat.T) / self.alignment_temp
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        loss = F.cross_entropy(similarity, labels)
        return loss

    def adversarial_loss(self, features, modality_labels):
        # 对抗训练损失
        logits = self.modality_discriminator(features)
        loss = F.cross_entropy(logits, modality_labels)
        return -loss  # Minimize the accuracy of the discriminator

    def forward(self, images, texts):
        img_semantic, img_scene = self.encode_image(images)
        txt_semantic, txt_scene = self.encode_text(texts)

        # Adversarial training step
        modality_labels = torch.tensor([0] * len(img_semantic) + [1] * len(txt_semantic)).to(img_semantic.device)
        all_features = torch.cat([img_semantic, txt_semantic], dim=0)
        adv_loss = self.adversarial_loss(all_features, modality_labels)

        features = {
            'img_semantic': img_semantic,
            'txt_semantic': txt_semantic,
            'img_scene': img_scene,
            'txt_scene': txt_scene,
            'graph': self.graph,
            'adv_loss': adv_loss
        }
        
        return features

def definition():
    return Definition()