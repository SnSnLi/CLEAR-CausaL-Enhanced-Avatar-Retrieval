from typing import Dict, Any
import networkx as nx
import numpy as np
import torch
from avatar.models.definition import Definition
import torch.nn.functional as F  

class CausalMediationAnalyzer:
    """
    Causal Mediation Analysis Class
    Implements mediation effect calculation based on graph structures to optimize model performance.
    """
    
    def __init__(self, definition: Definition):
        """
        Initialization function.
        :param definition: An instance of the Definition class, containing causal graph information.
        """
        self.definition = definition
        self.graph = definition.graph
        self.cmscm = definition.cmscm
        self.mediation_effects = None
        
    def calculate_mediation_effects(self) -> Dict[str, Any]:
        """
        Calculate mediation effects by integrating graph path analysis and CMSCM structural equations.
        :return: A dictionary containing mediation effect values.
        """
        if not isinstance(self.graph, nx.DiGraph):
            raise ValueError("Input graph must be a networkx.DiGraph")
            
        # 1. Graph path analysis
        graph_effects = {
            'direct': self._calculate_direct_effect(),
            'indirect': self._calculate_indirect_effect()
        }
        
        # 2. CMSCM structural equation analysis
        struct_effects = self.cmscm.compute_effects(
            self.definition.X,
            self.definition.Y
        )
        
        # 3. Dual-validation mechanism
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
        
        # 4. Combine results
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
        
    def _calculate_direct_effect(self) -> float:
        """
        Calculate the direct effect.
        """
        if 'image' not in self.graph or 'semantic' not in self.graph:
            return 0.0
            
        try:
            # Retrieve weights of direct paths
            direct_paths = list(nx.all_simple_paths(
                self.graph, 
                source='image', 
                target='semantic',
                cutoff=1  # Only consider direct paths
            ))
            
            if not direct_paths:
                return 0.0
                
            # Take the weight of the first direct path
            path = direct_paths[0]
            weights = []
            for u, v in zip(path[:-1], path[1:]):
                weights.append(self.graph[u][v].get('weight', 0.0))
                
            return np.prod(weights)
            
        except nx.NetworkXNoPath:
            return 0.0
            
    def _calculate_indirect_effect(self) -> float:
        """
        Calculate the indirect effect.
        """
        if 'image' not in self.graph or 'semantic' not in self.graph:
            return 0.0
            
        try:
            # Retrieve all indirect paths
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
                    weights.append(self.graph[u][v].get('weight', 0.0))
                    
                path_effect = np.prod(weights)
                total_effect += path_effect
                
            return total_effect
            
        except nx.NetworkXNoPath:
            return 0.0

    def optimize_model_weights(self) -> None:
        """
        Optimize model weights based on mediation effect analysis results.
        """
        if not self.mediation_effects:
            self.calculate_mediation_effects()
            
        effect_ratio = self.mediation_effects['effect_ratio']
        
        # Adjust parameters of the edge weight prediction network
        for param in self.definition.edge_weight_net.parameters():
            # Adjust learning rate based on mediation effect ratio
            param.requires_grad = True
            if effect_ratio > 0.5:  # If indirect effects dominate
                param.data *= (1 + effect_ratio * 0.1)  # Strengthen indirect path weights
            else:
                param.data *= (1 - effect_ratio * 0.1)  # Strengthen direct path weights

def causal_mediation_analysis(definition: Definition) -> Dict[str, float]:
    """
    Causal mediation analysis function.
    :param definition: An instance of the Definition class.
    :return: A dictionary containing mediation effect values.
    """
    analyzer = CausalMediationAnalyzer(definition)
    return analyzer.calculate_mediation_effects()
