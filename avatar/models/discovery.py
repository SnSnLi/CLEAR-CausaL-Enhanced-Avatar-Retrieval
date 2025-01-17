import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image
import spacy
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from avatar.models.definition import Definition
from avatar.models.analysis import CausalMediationAnalyzer

# 设置数据路径
annotations_directory = "/root/onethingai-tmp/avatar/data/flickr30k_entities/raw/annotations"
images_directory = "/root/onethingai-tmp/avatar/data/flickr30k_entities/raw/flickr30k_images"

class Relationship:
    def __init__(self, nlp_model=None, clip_model=None, clip_processor=None):
        self.nlp = nlp_model if nlp_model else spacy.load("en_core_web_sm")
        self.clip_model = clip_model if clip_model else CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = clip_processor if clip_processor else CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def parse_annotation(self, annotation_xml):
        root = ET.fromstring(annotation_xml)
        sentences = []

        for sentence_elem in root.findall('.//sentence'):
            sentence_id = sentence_elem.get('id')
            text = sentence_elem.text.strip()
            phrases = []

            for phrase_elem in sentence_elem.findall('.//phrase'):
                phrase_text = phrase_elem.find('text').text
                first_word_index = int(phrase_elem.find('first').text)
                last_word_index = int(phrase_elem.find('last').text)
                phrases.append({
                    'text': phrase_text,
                    'first': first_word_index,
                    'last': last_word_index
                })

            sentences.append({
                'id': sentence_id,
                'text': text,
                'phrases': phrases
            })

        return sentences

    def extract_causal_relationships(self, sentences, image_path=None):
        causal_relationships = defaultdict(float)

        for sentence in sentences:
            doc = self.nlp(sentence['text'])

            # 使用CLIP获取文本和图像的嵌入
            if image_path:
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = self.clip_processor(text=[sentence['text']], images=image, return_tensors="pt", padding=True)
                    outputs = self.clip_model(**inputs)
                    text_embedding = outputs.text_embeds
                    image_embedding = outputs.image_embeds

                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

            # 遍历依存句法树，寻找因果关系
            for token in doc:
                if token.dep_ == 'ROOT':  # 寻找句子的主要谓语动词
                    subject = None
                    obj = None
                    for child in token.children:
                        if child.dep_ == 'nsubj':  # 主语
                            subject = child.text
                        elif child.dep_ == 'dobj' or child.dep_ == 'pobj':  # 直接宾语或介词宾语
                            obj = child.text

                    if subject and obj:
                        relation = (subject, obj)
                        causal_relationships[relation] += 1  # 增加计数作为简单强度指标

        return dict(causal_relationships)

def process_annotations(xml_directory, image_directory, extractor):
    """遍历指定目录下的所有XML文件并提取因果关系"""
    all_causal_relationships = defaultdict(float)

    for filename in os.listdir(xml_directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(xml_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                annotation_xml = file.read()

            sentences = extractor.parse_annotation(annotation_xml)
            # 构建图像路径
            image_filename = os.path.splitext(filename)[0] + ".jpg"
            image_path = os.path.join(image_directory, image_filename) if os.path.exists(os.path.join(image_directory, image_filename)) else None

            relationships = extractor.extract_causal_relationships(sentences, image_path)
            for rel, count in relationships.items():
                all_causal_relationships[rel] += count

    return dict(all_causal_relationships)

class Discovery(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256):
        super().__init__()
        # 基础模型
        self.base_model = Definition(hidden_dim, shared_dim)
        
        # 分析器
        self.analyzer = CausalMediationAnalyzer(self.base_model)
        
        # 边权重网络
        self.edge_weight_net = nn.Sequential(
            nn.Linear(shared_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.base_model.parameters()) + 
            list(self.edge_weight_net.parameters()),
            lr=1e-4
        )
        
        # 对齐温度
        self.alignment_temp = 0.07

    def compute_alignment_loss(self, img_feat, txt_feat):
        # 计算对齐损失
        similarity = torch.matmul(img_feat, txt_feat.T) / self.alignment_temp
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        loss = F.cross_entropy(similarity, labels)
        return loss

    def update_edge_weights(self, features):
        """动态更新边权重"""
        for edge in features['graph'].edges():
            source, target = edge
            if source in features and target in features:
                combined = torch.cat([features[source], features[target]], dim=-1)
                weight = self.edge_weight_net(combined).squeeze(-1).item()
                features['graph'][source][target]['weight'] = weight

        # 根据因果关系提示进一步调整边权重
        self.update_causal_relationships(features)

    def update_causal_relationships(self, features):
        """根据因果关系提示更新边权重"""
        # 使用因果中介分析优化边权重
        analyzer = CausalMediationAnalyzer({
            'graph': features['graph'],
            'edge_weights': features['graph'].edges(data=True)
        })
        mediation_effects = analyzer.calculate_mediation_effects()
        
        # 根据中介效应调整边权重
        for (source, target), strength in self.causal_relationships.items():
            if (source, target) in features['graph'].edges():
                current_weight = features['graph'][source][target]['weight']
                # 根据中介效应比例调整权重
                updated_weight = min(
                    current_weight + strength * mediation_effects['effect_ratio'], 
                    1.0
                )
                features['graph'][source][target]['weight'] = updated_weight

    def forward(self, images, texts):
        # 1. 基础特征提取
        outputs = self.base_model(images, texts)
        
        # 2. CMSCM分析
        cmscm_outputs = self.base_model.cmscm.structural_equations(
            outputs['image'],
            outputs['text']
        )
        
        # 3. 因果效应分析
        mediation_effects = self.analyzer.calculate_mediation_effects(
            graph=outputs['graph'],
            edge_weights=outputs['graph'].edges(data=True)
        )
        
        # 4. 更新边权重
        self.update_edge_weights(outputs)
        weights = [data['weight'] for _, _, data in outputs['graph'].edges(data=True)]
        
        # 5. 合并所有输出
        return {
            **outputs,
            'cmscm_outputs': cmscm_outputs,
            'mediation_effects': mediation_effects,
            'edge_weights': weights,
            'direct_effects': mediation_effects['direct'],
            'indirect_effects': mediation_effects['indirect'],
            'total_effects': mediation_effects['total']
        }

    def discover_causal_relations(self, images, texts):
        outputs = self.forward(images, texts)
        
        # 结合图分析和CMSCM的发现
        discoveries = {
            'graph_paths': self._analyze_graph_paths(),
            'structural_relations': self._analyze_structural_relations(outputs),
            'effects': outputs['mediation_effects']
        }
        
        return discoveries

    def _analyze_structural_relations(self, outputs):
        cmscm_out = outputs['cmscm_outputs']
        return {
            'shared_semantics': cmscm_out['S'],
            'modality_specific': {
                'image': cmscm_out['Zx'],
                'text': cmscm_out['Zy']
            },
            'reconstruction': {
                'image': cmscm_out['X_hat'],
                'text': cmscm_out['Y_hat']
            }
        }

def discovery():
    """生成 discovery_output，包含动态计算的因果关系、边权重和中介效应"""
    extractor = Relationship()
    causal_relationships = process_annotations(annotations_directory, images_directory, extractor)
    
    model = Discovery(causal_relationships)
    
    # 返回动态计算的 discovery_output
    return model.get_discovery_output()

# 在模块加载时直接暴露 discovery_output
discovery_output = discovery()
