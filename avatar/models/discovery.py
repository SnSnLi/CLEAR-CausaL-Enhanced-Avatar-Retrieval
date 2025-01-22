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

# Set data paths
annotations_directory = "/root/onethingai-tmp/avatar/data/flickr30k_entities/raw/annotations"
images_directory = "/root/onethingai-tmp/avatar/data/flickr30k_entities/raw/flickr30k_images"

class Relationship:
    def __init__(self, nlp_model=None, clip_model=None, clip_processor=None):
        self.nlp = nlp_model if nlp_model else spacy.load("en_core_web_sm")
        self.clip_model = clip_model if clip_model else CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = clip_processor if clip_processor else CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def parse_annotation(self, annotation_xml):
        """
        Parse XML annotations to extract sentences and phrases.
        """
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
        """
        Extract causal relationships from sentences and image-text pairs.
        """
        causal_relationships = defaultdict(float)

        for sentence in sentences:
            doc = self.nlp(sentence['text'])

            # Use CLIP to get text and image embeddings
            if image_path:
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = self.clip_processor(text=[sentence['text']], images=image, return_tensors="pt", padding=True)
                    outputs = self.clip_model(**inputs)
                    text_embedding = outputs.text_embeds
                    image_embedding = outputs.image_embeds

                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

            # Traverse dependency tree to find causal relationships
            for token in doc:
                if token.dep_ == 'ROOT':  # Find the main verb of the sentence
                    subject = None
                    obj = None
                    for child in token.children:
                        if child.dep_ == 'nsubj':  # Subject
                            subject = child.text
                        elif child.dep_ == 'dobj' or child.dep_ == 'pobj':  # Direct or prepositional object
                            obj = child.text

                    if subject and obj:
                        relation = (subject, obj)
                        causal_relationships[relation] += 1  # Increment count as a simple strength metric

        return dict(causal_relationships)

def process_annotations(xml_directory, image_directory, extractor):
    """
    Process all XML files in the directory to extract causal relationships.
    """
    all_causal_relationships = defaultdict(float)

    for filename in os.listdir(xml_directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(xml_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                annotation_xml = file.read()

            sentences = extractor.parse_annotation(annotation_xml)
            # Build image path
            image_filename = os.path.splitext(filename)[0] + ".jpg"
            image_path = os.path.join(image_directory, image_filename) if os.path.exists(os.path.join(image_directory, image_filename)) else None

            relationships = extractor.extract_causal_relationships(sentences, image_path)
            for rel, count in relationships.items():
                all_causal_relationships[rel] += count

    return dict(all_causal_relationships)

class Discovery(nn.Module):
    def __init__(self, hidden_dim=768, shared_dim=256):
        super().__init__()
        # Base model
        self.base_model = Definition(hidden_dim, shared_dim)
        
        # Analyzer
        self.analyzer = CausalMediationAnalyzer(self.base_model)
        
        # Edge weight network
        self.edge_weight_net = nn.Sequential(
            nn.Linear(shared_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.base_model.parameters()) + 
            list(self.edge_weight_net.parameters()),
            lr=1e-4
        )
        
        # Alignment temperature
        self.alignment_temp = 0.07

    def compute_alignment_loss(self, img_feat, txt_feat):
        """
        Compute alignment loss between image and text features.
        """
        similarity = torch.matmul(img_feat, txt_feat.T) / self.alignment_temp
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        loss = F.cross_entropy(similarity, labels)
        return loss

    def update_edge_weights(self, features):
        """
        Dynamically update edge weights based on node importance and causal relationships.
        """
        # Update edge weights using the edge weight network
        for edge in features['graph'].edges():
            source, target = edge
            if source in features and target in features:
                combined = torch.cat([features[source], features[target]], dim=-1)
                weight = self.edge_weight_net(combined).squeeze(-1).item()
                features['graph'][source][target]['weight'] = weight

        # Further adjust edge weights based on causal relationships
        self.update_causal_relationships(features)

    def update_causal_relationships(self, features):
        """
        Update edge weights based on causal relationships and node importance.
        """
        # Use causal mediation analysis to optimize edge weights
        analyzer = CausalMediationAnalyzer({
            'graph': features['graph'],
            'edge_weights': features['graph'].edges(data=True)
        })
        mediation_effects = analyzer.calculate_mediation_effects()
        
        # Adjust edge weights based on mediation effects and node importance
        for (source, target), strength in self.causal_relationships.items():
            if (source, target) in features['graph'].edges():
                current_weight = features['graph'][source][target]['weight']
                # Adjust weight based on mediation effect ratio and node importance
                updated_weight = min(
                    current_weight + strength * mediation_effects['effect_ratio'] * features['graph'].nodes[source].get('importance', 1.0), 
                    1.0
                )
                features['graph'][source][target]['weight'] = updated_weight

    def forward(self, images, texts):
        """
        Forward pass with node importance and edge weight updates.
        """
        # 1. Base feature extraction
        outputs = self.base_model(images, texts)
        
        # 2. CMSCM analysis
        cmscm_outputs = self.base_model.cmscm.structural_equations(
            outputs['image'],
            outputs['text']
        )
        
        # 3. Causal effect analysis
        mediation_effects = self.analyzer.calculate_mediation_effects(
            graph=outputs['graph'],
            edge_weights=outputs['graph'].edges(data=True)
        )
        
        # 4. Update edge weights with node importance
        self.update_edge_weights(outputs)
        weights = [data['weight'] for _, _, data in outputs['graph'].edges(data=True)]
        
        # 5. Perform counterfactual analysis
        counterfactual_results = self.base_model.counterfactual_analysis(images, texts)
        
        # 6. Combine all outputs
        return {
            **outputs,
            'cmscm_outputs': cmscm_outputs,
            'mediation_effects': mediation_effects,
            'edge_weights': weights,
            'direct_effects': mediation_effects['direct'],
            'indirect_effects': mediation_effects['indirect'],
            'total_effects': mediation_effects['total'],
            'counterfactual_results': counterfactual_results
        }

    def discover_causal_relations(self, images, texts):
        """
        Discover causal relations using graph analysis and CMSCM.
        """
        outputs = self.forward(images, texts)
        
        # Combine graph analysis and CMSCM discoveries
        discoveries = {
            'graph_paths': self._analyze_graph_paths(),
            'structural_relations': self._analyze_structural_relations(outputs),
            'effects': outputs['mediation_effects'],
            'counterfactual_results': outputs['counterfactual_results']
        }
        
        return discoveries

    def _analyze_structural_relations(self, outputs):
        """
        Analyze structural relations from CMSCM outputs.
        """
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
    """
    Generate discovery_output with dynamically computed causal relationships, edge weights, and mediation effects.
    """
    extractor = Relationship()
    causal_relationships = process_annotations(annotations_directory, images_directory, extractor)
    
    model = Discovery(causal_relationships)
    
    # Return dynamically computed discovery_output
    return model.get_discovery_output()

# Expose discovery_output when the module is loaded
discovery_output = discovery()
