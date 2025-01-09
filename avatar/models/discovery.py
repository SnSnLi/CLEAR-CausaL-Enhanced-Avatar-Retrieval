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

# 设置数据路径
annotations_directory = "root/onethingai-tmp/avatar/data/flickr30k_entities/raw/annotations"
images_directory = "root/onethingai-tmp/avatar/data/flickr30k_entities/raw/flickr30k_images"

class Discovery(nn.Module):
    def __init__(self, causal_relationships, hidden_dim=768, shared_dim=256, lr=1e-4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared_dim = shared_dim
        
       
        self.base_model = Definition(hidden_dim, shared_dim)

        # 定义损失函数和优化器
        self.alignment_temp = 0.07
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=lr)

        # 边权重预测网络
        self.edge_weight_net = nn.Sequential(
            nn.Linear(shared_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出一个介于0到1之间的权重值
        )

        # 初始化因果关系
        self.causal_relationships = causal_relationships

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
        output = self.base_model(images, texts)
        img_semantic = output['img_semantic']
        txt_semantic = output['txt_semantic']

        # 计算对齐损失
        semantic_loss = self.compute_alignment_loss(img_semantic, txt_semantic)

        # 更新边权重
        self.update_edge_weights(output)
        weights = [data['weight'] for _, _, data in output['graph'].edges(data=True)]

        output.update({
            'semantic_loss': semantic_loss,
            'edge_weights': weights
        })

        if 'semantic_loss' in output:
            del output['semantic_loss']
        
        return output

def discovery():
    
    extractor = CausalRelationshipExtractor()

    causal_relationships = process_annotations(annotations_directory, images_directory, extractor)
    
    return Discovery(causal_relationships)


if __name__ == "__main__":
    model = discovery()
