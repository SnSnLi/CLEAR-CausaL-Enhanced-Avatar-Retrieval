import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from transformers import CLIPModel, CLIPTokenizer
import numpy as np
from typing import Dict, Tuple
from .cmscm import CausalVariable, CMSCM

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for computing node importance.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Final linear transformation
        output = self.out(output)
        return output, attn_weights

class Definition(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256, alignment_temp=0.07):
        super().__init__()
        # Preserve the original components
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # Modality disentangler for image and text
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
        
        # Multi-head attention for node importance computation
        self.multi_head_attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=8)
        
        # Modality suppression and gating
        self.modality_suppression = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Semantic projection and scene decomposition
        self.semantic_projection = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        self.scene_decomposer = HierarchicalSceneDecomposer(shared_dim, hidden_dim)
        
        # CM-SCM variable processing modules
        self.X = CausalVariable(hidden_dim)  # Image modality
        self.Y = CausalVariable(hidden_dim)  # Text modality
        self.S = CausalVariable(shared_dim)  # Shared semantics
        
        self.cmscm = CMSCM(hidden_dim, shared_dim)
        
        # Initialize the causal graph
        self.graph = nx.DiGraph()
        self._initialize_graph()
        
        # Edge weight prediction network
        self.edge_weight_net = nn.Sequential(
            nn.Linear(shared_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Alignment temperature and modality discriminator
        self.alignment_temp = alignment_temp
        self.modality_discriminator = ModalityDiscriminator(hidden_dim)

        # Scene analysis parameters
        self.similarity_threshold = 0.8  # Threshold for semantic similarity
        self.semantic_importance_scale = 1.5  # Scaling factor for semantic node importance

    def _initialize_graph(self):
        """
        Initialize the causal graph with nodes and base edges.
        """
        nodes = ['image', 'text', 'semantic', 'scene']
        self.graph.add_nodes_from(nodes)
        base_edges = [
            ('image', 'semantic'),
            ('text', 'semantic'),
            ('semantic', 'scene')
        ]
        self.graph.add_edges_from(base_edges)
        
    def compute_semantic_similarity(self, img_semantic, txt_semantic):
        """
        Compute the cosine similarity between image and text semantic features.
        """
        similarity = F.cosine_similarity(img_semantic, txt_semantic, dim=-1)
        return similarity.mean().item()  # Return the average similarity

    def adjust_node_importance_by_scene(self, img_semantic, txt_semantic):
        """
        Adjust node importance based on scene analysis (semantic similarity).
        """
        similarity = self.compute_semantic_similarity(img_semantic, txt_semantic)
        
        # If semantic similarity is high, increase the importance of the 'semantic' node
        if similarity > self.similarity_threshold:
            self.graph.nodes['semantic']['importance'] *= self.semantic_importance_scale
            print(f"High semantic similarity ({similarity:.2f}), increasing 'semantic' node importance")
        else:
            print(f"Low semantic similarity ({similarity:.2f}), keeping 'semantic' node importance")

    def compute_node_importance_with_attention(self, features):
        """
        Compute node importance using multi-head attention.
        """
        # Stack all node features for attention computation
        node_features = torch.stack([features[node] for node in self.graph.nodes], dim=1)
        
        # Compute attention weights using multi-head attention
        _, attn_weights = self.multi_head_attention(node_features, node_features, node_features)
        
        # Compute node importance as the mean attention weight across heads and batches
        node_importance = attn_weights.mean(dim=1).mean(dim=1)
        
        # Assign importance scores to each node in the graph
        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['importance'] = node_importance[i].item()
        
        print("Node importance scores:", {node: self.graph.nodes[node]['importance'] for node in self.graph.nodes})

    def process_features(self, x, modality):
        """
        Process features by disentangling modalities and adding noise.
        """
        # Disentangle modality-specific features
        x_disentangled = self.disentangle_modality(x, modality)
        x_suppressed = self.suppress_modality(x_disentangled)
        
        # Add noise using CM-SCM
        if modality == 'image':
            x_noised = self.X.add_noise(x_suppressed)
        else:
            x_noised = self.Y.add_noise(x_suppressed)
            
        return x_noised, x_suppressed

    def compute_structural_equations(self, img_feat, txt_feat):
        """
        Compute structural equations using CM-SCM.
        """
        return self.cmscm.structural_equations(img_feat, txt_feat)

    def encode_image(self, image):
        """
        Enhanced image encoding with noise and semantic projection.
        """
        img_feat = self.clip.get_image_features(image)
        img_noised, img_suppressed = self.process_features(img_feat, 'image')
        img_semantic = self.semantic_projection(img_suppressed)
        img_scene = self.scene_decomposer(img_semantic)
        return img_semantic, img_scene, img_noised

    def encode_text(self, text):
        """
        Enhanced text encoding with noise and semantic projection.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        txt_feat = self.clip.get_text_features(**inputs)
        txt_noised, txt_suppressed = self.process_features(txt_feat, 'text')
        txt_semantic = self.semantic_projection(txt_suppressed)
        txt_scene = self.scene_decomposer(txt_semantic)
        return txt_semantic, txt_scene, txt_noised

    def compute_mediation_effects(self, features):
        """
        Compute mediation effects using both graph path analysis and structural equations.
        """
        effects = {}
        
        # Variable mapping
        X = features['img_noised']  # Image modality
        Y = features['txt_noised']  # Text modality
        S = features['semantic']    # Shared semantics
        Zx = features['img_semantic']  # Image-specific semantics
        Zy = features['txt_semantic']  # Text-specific semantics
        
        # 1. Graph path analysis
        for edge in self.graph.edges():
            source, target = edge
            if source in features and target in features:
                combined = torch.cat([features[source], features[target]], dim=-1)
                weight = self.edge_weight_net(combined)
                self.graph[source][target]['weight'] = weight.item()
                
        # 2. Structural equation computation
        struct_outputs = self.compute_structural_equations(X, Y)
        
        # 3. Dual verification mechanism
        # 3.1 Graph path effects
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
        
        # 3.2 Structural equation effects
        structural_effects = {
            'S': struct_outputs['S'],
            'Zx': struct_outputs['Zx'],
            'Zy': struct_outputs['Zy'],
            'R': struct_outputs['R']
        }
        
        # 4. Consistency verification
        consistency_loss = F.mse_loss(
            torch.tensor([
                graph_effects['direct']['img_to_semantic'],
                graph_effects['direct']['txt_to_semantic'],
                graph_effects['indirect']['img_to_scene'],
                graph_effects['indirect']['txt_to_scene']
            ]),
            torch.tensor([
                structural_effects['S'].mean().item(),
                structural_effects['R'].mean().item(),
                structural_effects['Zx'].mean().item(),
                structural_effects['Zy'].mean().item()
            ])
        )
        
        # 5. Return integrated results
        effects.update({
            'graph_effects': graph_effects,
            'structural_effects': structural_effects,
            'consistency_loss': consistency_loss.item()
        })
        
        return effects

    def forward(self, images, texts):
        """
        Enhanced forward pass with scene analysis and attention-based node importance.
        """
        # 1. Feature encoding
        img_semantic, img_scene, img_noised = self.encode_image(images)
        txt_semantic, txt_scene, txt_noised = self.encode_text(texts)

        # 2. Scene analysis: Adjust node importance based on semantic similarity
        self.adjust_node_importance_by_scene(img_semantic, txt_semantic)

        # 3. Build feature dictionary
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

        # 4. Compute node importance using multi-head attention
        self.compute_node_importance_with_attention(features)

        # 5. Compute mediation effects
        mediation_effects = self.compute_mediation_effects(features)

        # 6. Return results
        return {
            **features,
            'effects': mediation_effects,
            'graph': self.graph,
            'adv_loss': adv_loss
        }

def definition():
    return Definition()