from typing import Dict, Any
import networkx as nx
import numpy as np
import torch
from avatar.models.definition import Definition
import torch.nn.functional as F

class CausalMediationAnalyzer:
    """
    Causal Mediation Analyzer class.
    Implements mediation effect calculation based on graph structure and node importance.
    """

    def __init__(self, definition: Definition):
        """
        Initializes the analyzer.
        :param definition: An instance of the Definition class containing the causal graph and other information.
        """
        self.definition = definition
        self.graph = definition.graph
        self.cmscm = definition.cmscm
        self.mediation_effects = None

    def calculate_mediation_effects(self) -> Dict[str, Any]:
        """
        Calculates mediation effects by integrating graph path analysis and CMSCM structural equations.
        :return: A dictionary containing mediation effect values.
        """
        if not isinstance(self.graph, nx.DiGraph):
            raise ValueError("Input graph must be a networkx.DiGraph")

        # 1. Compute node importance using scene analysis and attention mechanism
        self._compute_node_importance()

        # 2. Graph path analysis
        graph_effects = {
            'direct': self._calculate_direct_effect(),
            'indirect': self._calculate_indirect_effect()
        }

        # 3. CMSCM structural equation analysis
        struct_effects = self.cmscm.compute_effects(
            self.definition.X,
            self.definition.Y
        )

        # 4. Dual verification mechanism
        consistency_loss = F.mse_loss(
            torch.tensor([
                graph_effects['direct'],
                graph_effects['indirect']
            ]),
            torch.tensor([
                struct_effects['direct'],
                struct_effects['indirect']
            ])
        )

        # 5. Combine results
        total_effect = (
            graph_effects['direct'] +
            graph_effects['indirect'] +
            struct_effects['direct'] +
            struct_effects['indirect']
        )

        self.mediation_effects = {
            'graph_effects': graph_effects,
            'struct_effects': struct_effects,
            'total_effect': total_effect,
            'consistency_loss': consistency_loss.item(),
            'effect_ratio': (graph_effects['indirect'] + struct_effects['indirect']) / total_effect
        }

        return self.mediation_effects

    def _compute_node_importance(self):
        """
        Computes node importance using scene analysis and attention mechanisms.
        """
        # Scene analysis: Adjust node importance based on semantic similarity
        img_semantic = self.definition.encode_image(self.definition.X)
        txt_semantic = self.definition.encode_text(self.definition.Y)
        self.definition.adjust_node_importance_by_scene(img_semantic, txt_semantic)

        # Attention mechanism: Compute node importance using multi-head attention
        features = {
            'image': img_semantic,
            'text': txt_semantic,
            'semantic': (img_semantic + txt_semantic) / 2,
            'scene': self.definition.scene_decomposer((img_semantic + txt_semantic) / 2)
        }
        self.definition.compute_node_importance_with_attention(features)

    def _calculate_direct_effect(self) -> float:
        """
        Calculates the direct effect in the causal graph.
        """
        if 'image' not in self.graph or 'semantic' not in self.graph:
            return 0.0

        try:
            # Get direct paths and their weights
            direct_paths = list(nx.all_simple_paths(
                self.graph,
                source='image',
                target='semantic',
                cutoff=1  # Only consider direct paths
            ))

            if not direct_paths:
                return 0.0

            # Take the first direct path and compute its weighted effect
            path = direct_paths[0]
            weights = []
            for u, v in zip(path[:-1], path[1:]):
                weights.append(self.graph[u][v].get('weight', 0.0) * self.graph.nodes[u].get('importance', 1.0))

            return np.prod(weights)

        except nx.NetworkXNoPath:
            return 0.0

    def _calculate_indirect_effect(self) -> float:
        """
        Calculates the indirect effect in the causal graph.
        """
        if 'image' not in self.graph or 'semantic' not in self.graph:
            return 0.0

        try:
            # Get all indirect paths
            indirect_paths = list(nx.all_simple_paths(
                self.graph,
                source='image',
                target='semantic',
                cutoff=len(self.graph)  # Consider all possible paths
            ))

            if not indirect_paths:
                return 0.0

            total_effect = 0.0
            for path in indirect_paths:
                if len(path) <= 2:  # Skip direct paths
                    continue

                weights = []
                for u, v in zip(path[:-1], path[1:]):
                    weights.append(self.graph[u][v].get('weight', 0.0) * self.graph.nodes[u].get('importance', 1.0))

                path_effect = np.prod(weights)
                total_effect += path_effect

            return total_effect

        except nx.NetworkXNoPath:
            return 0.0

    def optimize_model_weights(self) -> None:
        """
        Optimizes model weights based on mediation effect analysis.
        """
        if not self.mediation_effects:
            self.calculate_mediation_effects()

        effect_ratio = self.mediation_effects['effect_ratio']

        # Adjust edge weight prediction network parameters
        for param in self.definition.edge_weight_net.parameters():
            # Adjust learning rate based on mediation effect ratio
            param.requires_grad = True
            if effect_ratio > 0.5:  # If indirect effect dominates
                param.data *= (1 + effect_ratio * 0.1)  # Strengthen indirect path weights
            else:
                param.data *= (1 - effect_ratio * 0.1)  # Strengthen direct path weights

def causal_mediation_analysis(definition: Definition) -> Dict[str, float]:
    """
    Performs causal mediation analysis.
    :param definition: An instance of the Definition class.
    :return: A dictionary containing mediation effect values.
    """
    analyzer = CausalMediationAnalyzer(definition)
    return analyzer.calculate_mediation_effects()