import os
import os.path as osp
from typing import Dict, Any, List
from avatar.models.analysis import causal_mediation_analysis
from avatar.models.causal_score import CausalScoreCalculator
from avatar.prompts.discovery import discovery_output

class PromptGenerator:
    def __init__(self, discovery_output: Dict[str, Any]):
        """
        Initialize the prompt generator with discovery output
        
        Args:
            discovery_output: Output from discovery module containing causal relationships
        """
        self.discovery_output = discovery_output
        self.template_backup = {}

    def _backup_template(self, template_path: str) -> None:
        """
        Backup the original template content.
        
        Args:
            template_path: Path to the template file.
        """
        with open(template_path, 'r') as f:
            self.template_backup[template_path] = f.read()

    def _restore_template(self, template_path: str) -> None:
        """
        Restore the original template content.
        
        Args:
            template_path: Path to the template file.
        """
        if template_path in self.template_backup:
            with open(template_path, 'w') as f:
                f.write(self.template_backup[template_path])

    def _extract_causal_insights(self) -> Dict[str, Any]:
        
        insights = {}
        
        # Extract and process causal relationships with weights
        raw_relationships = self.discovery_output.get('causal_relationships', [])
        insights['causal_relationships'] = [
            {
                'cause': rel[0],
                'effect': rel[1],
                'strength': self.discovery_output['edge_weights'][i],
                'direct_effect': 0.0,
                'indirect_effect': 0.0,
                'counterfactual_effect': 0.0  # 新增反事实效应
            }
            for i, rel in enumerate(raw_relationships)
        ]
        
        # Calculate mediation effects and update relationship weights
        mediation_effects = causal_mediation_analysis(self.discovery_output)
        for rel in insights['causal_relationships']:
            cause = rel['cause']
            effect = rel['effect']
            
            # Update direct and indirect effects
            rel['direct_effect'] = mediation_effects.get('direct_effect', 0.0)
            rel['indirect_effect'] = mediation_effects.get('indirect_effect', 0.0)
            
            
            if 'counterfactual_results' in self.discovery_output:
                counterfactual_effects = self.discovery_output['counterfactual_results'].get('causal_effects', {})
                rel['counterfactual_effect'] = counterfactual_effects.get('total_effect', 0.0)
            
            # Adjust strength based on effect ratio
            effect_ratio = mediation_effects.get('effect_ratio', 0.5)
            rel['strength'] = rel['strength'] * (1 + node_importance[cause] + node_importance[effect])
            
        # Extract key variables with their causal importance
        insights['key_variables'] = self._identify_key_variables()
        insights['mediation_effects'] = mediation_effects
        
      
        if 'counterfactual_results' in self.discovery_output:
            insights['counterfactual_results'] = self.discovery_output['counterfactual_results']
        
        return insights

    def _identify_key_variables(self) -> List[str]:
        """
        Identify key variables based on centrality and total effect.
        
        Returns:
            List of key variable names.
        """
        key_variables = []
        
        # Calculate centrality scores (e.g., degree centrality)
        centrality_scores = self._calculate_centrality_scores()
        
        # Calculate total effect for each node
        total_effects = self._calculate_total_effects()
        
        # Combine centrality and total effect to identify key variables
        for node in centrality_scores:
            # Calculate a combined score (e.g., centrality * total effect)
            combined_score = centrality_scores[node] * total_effects.get(node, 1.0) * node_importance[node]
            
            # If the combined score is above a threshold, consider it a key variable
            if combined_score > 0.5:  # Adjust threshold as needed
                key_variables.append(node)
        
        return key_variables

    def _calculate_centrality_scores(self) -> Dict[str, float]:
        """
        Calculate centrality scores (e.g., degree centrality) for each node.
        
        Returns:
            Dictionary mapping node names to their centrality scores.
        """
        centrality_scores = {}
        
        # Count the number of relationships for each node (degree centrality)
        for rel in self.discovery_output.get('causal_relationships', []):
            cause, effect = rel[0], rel[1]
            centrality_scores[cause] = centrality_scores.get(cause, 0) + 1
            centrality_scores[effect] = centrality_scores.get(effect, 0) + 1
        
        # Normalize centrality scores to a range of 0 to 1
        max_centrality = max(centrality_scores.values(), default=1.0)
        for node in centrality_scores:
            centrality_scores[node] /= max_centrality
        
        return centrality_scores

    def _calculate_total_effects(self) -> Dict[str, float]:
        """
        Calculate total effect for each node based on causal relationships.
        
        Returns:
            Dictionary mapping node names to their total effects.
        """
        total_effects = {}
        
        # Sum the strength of all relationships involving each node
        for rel in self.discovery_output.get('causal_relationships', []):
            cause, effect = rel[0], rel[1]
            strength = self.discovery_output['edge_weights'][rel]
            total_effects[cause] = total_effects.get(cause, 0.0) + strength
            total_effects[effect] = total_effects.get(effect, 0.0) + strength
        
        return total_effects

    def calculate_causal_score(self, node: str, causal_insights: Dict[str, Any]) -> float:
        """
        Calculate the causal score for a node based on causal insights.
        
        Args:
            node: The node to calculate the score for.
            causal_insights: Dictionary containing causal insights (e.g., relationships, effects, key variables).
        
        Returns:
            The causal score for the node.
        """
        score = 0.0
        score += node_importance[node] * 0.5
        
        # 1. Check if the node is a key variable
        if node in causal_insights['key_variables']:
            score += 1.0  # Add a base score for being a key variable
        
        # 2. Sum the strength of all relationships involving the node
        for rel in causal_insights['causal_relationships']:
            if node in [rel['cause'], rel['effect']]:
                score += rel['strength']
        
        # 3. Add direct and indirect effects
        for rel in causal_insights['causal_relationships']:
            if node == rel['cause']:
                score += rel['direct_effect'] * 0.5  # Direct effect contributes to the score
                score += rel['indirect_effect'] * 0.3  # Indirect effect contributes less
        
        # 4. Add total effect if available
        if 'mediation_effects' in causal_insights:
            score += causal_insights['mediation_effects'].get('total_effect', 0.0) * 0.2
        
        # 5. Add centrality score
        centrality_scores = self._calculate_centrality_scores()
        score += centrality_scores.get(node, 0.0) * 0.5  # Centrality contributes to the score
        
        # 6. Normalize the score to a range of 0 to 1 (optional)
        score = min(1.0, max(0.0, score))
        
        return score

    def get_node_score_dict(self, query: str, candidate_ids: List[str], **parameter_dict) -> Dict[str, float]:
        """
        Generate a dictionary of node scores based on causal insights.
        
        Args:
            query: The user query.
            candidate_ids: List of candidate node IDs.
            parameter_dict: Additional parameters for scoring.
        
        Returns:
            Dictionary mapping node IDs to their scores.
        """
        node_score_dict = {}
        
        # Extract causal insights
        causal_insights = self._extract_causal_insights()
        
        # Calculate scores for each candidate node
        for node in candidate_ids:
            # Calculate the causal score for the node
            causal_score = self.calculate_causal_score(node, causal_insights)
            
            # Combine causal score with other factors (e.g., query relevance)
            final_score = causal_score * self._calculate_query_relevance(node, query)
            
            # Store the final score in the dictionary
            node_score_dict[node] = final_score
        
        return node_score_dict

    def _calculate_query_relevance(self, node: str, query: str) -> float:
        """
        Calculate the relevance of a node to the user query.
        
        Args:
            node: The node to calculate relevance for.
            query: The user query.
        
        Returns:
            The relevance score (0 to 1).
        """
        # Implement query relevance logic here
        # For example, check if the node name appears in the query
        return 1.0 if node.lower() in query.lower() else 0.5

    def generate_comparator_prompt(self, template_path: str, pos_neg_queries: str) -> str:
        """
        Generate comparator prompt with causal weights
        
        Args:
            template_path: Path to the original comparator template
            pos_neg_queries: Positive and negative examples to include in prompt
        
        Returns:
            Generated prompt string with causal weights
        """
        # Load base template
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Extract causal insights with weights
        causal_insights = self._extract_causal_insights()
        
        # Generate weighted discovery section
        discovery_section = self._generate_discovery_section(causal_insights)
        
        # Generate causal reasoning guidelines
        reasoning_section = self._generate_reasoning_section(causal_insights)
        
        # Combine sections with template
        prompt = template.replace('<pos_neg_queries>', pos_neg_queries)
        prompt = prompt.replace('<discovery_insights>', discovery_section)
        prompt = prompt.replace('<reasoning_guidelines>', reasoning_section)
        
        # Add causal weight summary
        total_effect = causal_insights['mediation_effects']['total_effect']
        effect_ratio = causal_insights['mediation_effects']['effect_ratio']
        prompt += f"\n### Causal Weight Summary\n"
        prompt += f"- Total Effect: {total_effect:.2f}\n"
        prompt += f"- Effect Ratio: {effect_ratio:.2f}\n"
        
        return prompt

    def generate_avatar_initialize_prompt(self, template_path: str) -> str:
        """
        Generate avatar initialize prompt with causal scores
        
        Args:
            template_path: Path to the original template
        
        Returns:
            Generated prompt string with causal scores
        """
        # Load base template
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Extract causal insights
        causal_insights = self._extract_causal_insights()
        
        # Update template with causal scores
        prompt = template.replace('<causal_score>', f"{causal_insights['causal_score']:.2f}")
        
        # Return the generated prompt
        return prompt

    def generate_improve_actions_prompt(self, template_path: str, feedback: str) -> str:
        """
        Generate improve actions prompt with causal guidance
        
        Args:
            template_path: Path to the original template
            feedback: Feedback message to include in prompt
        
        Returns:
            Generated prompt string with causal improvement guidance
        """
        # Load base template
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Extract causal insights with weights
        causal_insights = self._extract_causal_insights()
        
        # Generate weighted discovery section
        discovery_section = self._generate_discovery_section(causal_insights)
        
        # Generate causal reasoning guidelines
        reasoning_section = self._generate_reasoning_section(causal_insights)
        
        # Generate improvement suggestions based on causal weights
        improvement_section = "### Causal Improvement Suggestions\n"
        for rel in causal_insights['causal_relationships']:
            if rel['direct_effect'] > rel['indirect_effect']:
                improvement_section += (
                    f"- Strengthen direct path from {rel['cause']} to {rel['effect']} "
                    f"(current strength: {rel['strength']:.2f})\n"
                )
            else:
                improvement_section += (
                    f"- Optimize mediator variables between {rel['cause']} and {rel['effect']} "
                    f"(current indirect effect: {rel['indirect_effect']:.2f})\n"
                )
        
        # Add a reminder to include causal scores in get_node_score_dict
        causal_score_reminder = (
            "\n### Important Reminder\n"
            "When modifying the `get_node_score_dict` function, ensure that you incorporate the causal scores "
            "into the node scoring logic. The causal scores can be used to prioritize nodes based on their "
            "causal importance and the strength of their relationships. For example:\n"
            "```python\n"
            "def get_node_score_dict(query, candidate_ids, **parameter_dict):\n"
            "    node_score_dict = {}\n"
            "    for node in candidate_ids:\n"
            "        # Calculate the causal score for the node based on causal insights\n"
            "        causal_score = calculate_causal_score(node, causal_insights)\n"
            "        # Combine causal score with other scoring factors\n"
            "        node_score_dict[node] = causal_score * other_factors(node)\n"
            "    return node_score_dict\n"
            "```\n"
            "Make sure to define `calculate_causal_score` to compute the causal importance of each node.\n"
        )
        
        # Combine sections with template
        prompt = template.replace('<feedback_message>', feedback)
        prompt = prompt.replace('<discovery_insights>', discovery_section)
        prompt = prompt.replace('<reasoning_guidelines>', reasoning_section)
        prompt = prompt.replace('<improvement_suggestions>', improvement_section)
        prompt += causal_score_reminder  # Add the reminder to the prompt
        
        return prompt

   def _generate_discovery_section(self, insights: Dict[str, Any]) -> str:
        section = "### Discovery Insights with Causal Weights\n"
        section += "Based on the causal analysis, the following key insights were discovered:\n"
        
        # Add weighted causal relationships
        if insights['causal_relationships']:
            section += "- Weighted Causal Relationships:\n"
            for rel in insights['causal_relationships']:
                section += (
                    f"  * {rel['cause']} → {rel['effect']} "
                    f"(strength: {rel['strength']:.2f}, "
                    f"direct: {rel['direct_effect']:.2f}, "
                    f"indirect: {rel['indirect_effect']:.2f}, "
                    f"counterfactual: {rel.get('counterfactual_effect', 0.0):.2f})\n"  # 新增反事实效应
                )
        
        # Add mediation effects with weights
        if insights['mediation_effects']:
            section += "- Mediation Effects with Weights:\n"
            section += (
                f"  * Direct Effect: {insights['mediation_effects']['direct_effect']:.2f}\n"
                f"  * Indirect Effect: {insights['mediation_effects']['indirect_effect']:.2f}\n"
                f"  * Total Effect: {insights['mediation_effects']['total_effect']:.2f}\n"
                f"  * Effect Ratio: {insights['mediation_effects']['effect_ratio']:.2f}\n"
            )
        
        # 新增：添加反事实分析结果
        if 'counterfactual_results' in insights:
            section += "- Counterfactual Analysis Results:\n"
            counterfactual_effects = insights['counterfactual_results'].get('causal_effects', {})
            section += (
                f"  * Total Counterfactual Effect: {counterfactual_effects.get('total_effect', 0.0):.2f}\n"
            )
        
        # Add key variables with causal importance       
        if insights['key_variables']:
            section += "- Key Variables with Causal Importance:\n"
            for var in insights['key_variables']:
                # Calculate importance score based on connected relationships and node importance
                relationship_importance = sum(
                    rel['strength'] for rel in insights['causal_relationships']
                    if var in [rel['cause'], rel['effect']]
                )
                # Combine relationship importance with node importance
                total_importance = relationship_importance * node_importance.get(var, 1.0)
                section += f"  * {var} (importance: {total_importance:.2f})\n"
        
        return section

    def _generate_reasoning_section(self, insights: Dict[str, Any]) -> str:
    section = "### Causal Reasoning Guidelines\n"
    section += "When comparing actions, consider the following causal factors:\n"
    
    # Add weighted causal reasoning
    if insights['causal_relationships']:
        section += "- Weighted Causal Reasoning:\n"
        for rel in insights['causal_relationships']:
            if rel['direct_effect'] > rel['indirect_effect']:
                reasoning = (
                    f"  * Direct path from {rel['cause']} to {rel['effect']} is stronger "
                    f"(direct: {rel['direct_effect']:.2f} > indirect: {rel['indirect_effect']:.2f})\n"
                )
            else:
                reasoning = (
                    f"  * Indirect path from {rel['cause']} to {rel['effect']} is stronger "
                    f"(indirect: {rel['indirect_effect']:.2f} > direct: {rel['direct_effect']:.2f})\n"
                )
            section += reasoning
    
    
    if 'counterfactual_results' in insights:
        counterfactual_effects = insights['counterfactual_results'].get('causal_effects', {})
        section += "- Counterfactual Reasoning:\n"
        section += (
            f"  * If certain variables were changed, the total effect would be {counterfactual_effects.get('total_effect', 0.0):.2f}\n"
        )
    
    # Add mediation effect guidance
    if insights['mediation_effects']:
        effect_ratio = insights['mediation_effects']['effect_ratio']
        if effect_ratio > 0.5:
            section += (
                f"- Indirect effects dominate (ratio: {effect_ratio:.2f}), "
                "focus on mediator variables\n"
            )
        else:
            section += (
                f"- Direct effects dominate (ratio: {1 - effect_ratio:.2f}), "
                "focus on direct causal paths\n"
            )
    
    # Add key variable guidance with weights      
    if insights['key_variables']:
        section += "- Key Variable Prioritization:\n"
        # Sort key variables by node importance and relationship strength
        sorted_vars = sorted(
            insights['key_variables'],
            key=lambda var: (
                node_importance.get(var, 1.0) *  # Node importance
                sum(  # Relationship strength
                    rel['strength'] for rel in insights['causal_relationships']
                    if var in [rel['cause'], rel['effect']]
                )
            ),
            reverse=True
        )
        for var in sorted_vars:
            # Calculate importance based on both node importance and relationship strength
            importance = (
                node_importance.get(var, 1.0) *
                sum(
                    rel['strength'] for rel in insights['causal_relationships']
                    if var in [rel['cause'], rel['effect']]
                )
            )
            section += f"  * {var} (importance: {importance:.2f})\n"
    
    return section
        
        return section
