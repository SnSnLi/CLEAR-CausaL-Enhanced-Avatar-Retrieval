import torch
import numpy as np
from typing import Dict, Any
from .analysis import CausalMediationAnalyzer
from .discovery import Discovery

class CausalScoreCalculator:
    """
    Causal Score Calculator
    Integrates all relevant weight calculations, including:
    - Direct effect weight
    - Indirect effect weight
    - Image quality weight
    - Phrase relevance weight
    - Semantic embedding similarity
    - Graph structure weight
    - Semantic alignment loss
    - Counterfactual effect weight
    """
    
    def __init__(self, discovery: Discovery):
        """
        Initialize the calculator.
        :param discovery: Discovery instance containing all relevant weight information.
        """
        self.discovery = discovery
        self.analyzer = CausalMediationAnalyzer(discovery)
        
        # Initialize weight parameters
        self.direct_effect_weight = 0.3
        self.indirect_effect_weight = 0.4
        self.image_quality_weight = 0.2
        self.phrase_relevance_weight = 0.3
        self.semantic_similarity_weight = 0.4
        self.graph_structure_weight = 0.3
        self.semantic_loss_weight = 0.2
        self.counterfactual_weight = 0.5  # New weight for counterfactual analysis
        
        # Initialize CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip.eval()
        
    def _calculate_image_quality(self, image: torch.Tensor) -> float:
        """
        Calculate image quality score based on CLIP features.
        """
        with torch.no_grad():
            img_feat = self.clip.get_image_features(image)
            l2_norm = torch.norm(img_feat, p=2)
            mean_feat = img_feat.mean(dim=0)
            cosine_sim = F.cosine_similarity(img_feat, mean_feat.unsqueeze(0))
            return 0.7 * l2_norm + 0.3 * cosine_sim

    def _calculate_phrase_relevance(self, phrase: str, context: str) -> float:
        """
        Calculate phrase relevance based on CLIP text encoding.
        """
        with torch.no_grad():
            phrase_input = self.tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
            context_input = self.tokenizer(context, return_tensors="pt", padding=True, truncation=True)
            
            phrase_feat = self.clip.get_text_features(**phrase_input)
            context_feat = self.clip.get_text_features(**context_input)
            
            return F.cosine_similarity(phrase_feat, context_feat).item()

    def calculate_causal_score(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate the comprehensive causal score.
        :param output: Discovery output dictionary.
        :return: Dictionary containing all weights and the final score.
        """
        # Calculate image quality and phrase relevance
        if 'image' in output:
            output['image_quality'] = self._calculate_image_quality(output['image'])
        if 'phrase' in output and 'context' in output:
            output['phrase_relevance'] = self._calculate_phrase_relevance(
                output['phrase'], 
                output['context']
            )
            
        # Get mediation effects
        mediation_effects = self.analyzer.calculate_mediation_effects()
        
        # Calculate semantic embedding similarity
        semantic_similarity = self._calculate_semantic_similarity(
            output['img_semantic'],
            output['txt_semantic']
        )
        
        # Calculate graph structure score with node importance
        graph_structure_score = self._calculate_graph_structure_score(
            output['graph'],
            output['edge_weights']
        )
        
        # Calculate counterfactual effect score
        counterfactual_score = self._calculate_counterfactual_score(
            output.get('counterfactual_results', {})
        )
        
        # Calculate individual scores
        direct_score = mediation_effects['direct_effect'] * self.direct_effect_weight
        indirect_score = mediation_effects['indirect_effect'] * self.indirect_effect_weight
        image_quality_score = output.get('image_quality', 1.0) * self.image_quality_weight
        phrase_relevance_score = output.get('phrase_relevance', 1.0) * self.phrase_relevance_weight
        semantic_similarity_score = semantic_similarity * self.semantic_similarity_weight
        graph_structure_score = graph_structure_score * self.graph_structure_weight
        semantic_loss_score = output['semantic_loss'] * self.semantic_loss_weight
        counterfactual_score = counterfactual_score * self.counterfactual_weight  # Add counterfactual score
        
        # Calculate the comprehensive causal score
        causal_score = (
            direct_score + 
            indirect_score + 
            image_quality_score + 
            phrase_relevance_score +
            semantic_similarity_score +
            graph_structure_score -
            semantic_loss_score +  # Subtract loss value
            counterfactual_score  # Add counterfactual effect
        )
        
        return {
            'causal_score': causal_score,
            'direct_effect': direct_score,
            'indirect_effect': indirect_score,
            'image_quality': image_quality_score,
            'phrase_relevance': phrase_relevance_score,
            'semantic_similarity': semantic_similarity_score,
            'graph_structure': graph_structure_score,
            'semantic_loss': semantic_loss_score,
            'counterfactual_score': counterfactual_score,  # Add counterfactual score
            'mediation_effects': mediation_effects
        }

    def _calculate_semantic_similarity(self, img_semantic: torch.Tensor, txt_semantic: torch.Tensor) -> float:
        """
        Calculate semantic embedding similarity.
        """
        return torch.cosine_similarity(img_semantic, txt_semantic, dim=-1).mean().item()

    def _calculate_graph_structure_score(self, graph: Any, edge_weights: list) -> float:
        """
        Calculate graph structure score with node importance.
        """
        if not edge_weights:
            return 0.0
        # Incorporate node importance into the graph structure score
        node_importance = [graph.nodes[node].get('importance', 1.0) for node in graph.nodes]
        return np.mean(edge_weights) * np.mean(node_importance)

    def _calculate_counterfactual_score(self, counterfactual_results: Dict[str, Any]) -> float:
        """
        Calculate counterfactual effect score based on counterfactual analysis results.
        """
        if not counterfactual_results:
            return 0.0
        
        # Use the total effect from counterfactual analysis as the score
        total_effect = counterfactual_results.get('causal_effects', {}).get('total_effect', 0.0)
        return total_effect

    def update_weights(self, **kwargs) -> None:
        """
        Dynamically update weight parameters.
        :param kwargs: Weight parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid weight parameter: {key}")

def calculate_causal_score(discovery: Discovery, output: Dict[str, Any]) -> Dict[str, float]:
    """
    Causal score calculation function.
    :param discovery: Discovery instance.
    :param output: Discovery output dictionary.
    :return: Dictionary containing all weights and the final score.
    """
    calculator = CausalScoreCalculator(discovery)
    return calculator.calculate_causal_score(output)
