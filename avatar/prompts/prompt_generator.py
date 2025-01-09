import os
import os.path as osp
from typing import Dict, Any
from typing import List
from avatar.models.analysis import causal_mediation_analysis

class PromptGenerator:
    def __init__(self, discovery_output: Dict[str, Any]):
        """
        Initialize the prompt generator with discovery output
        
        Args:
            discovery_output: Output from discovery module containing causal relationships
        """
        self.discovery_output = discovery_output
        
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
            Path to the generated prompt file
        """
        # Load base template
        with open(template_path, 'r') as f:
            template = f.read()
            
        # Extract causal insights
        causal_insights = self._extract_causal_insights()
        
        # Update template with causal scores
        template = template.replace('<causal_score>', f"{causal_insights['causal_score']:.2f}")
        
        # Write new prompt file
        output_path = osp.join(osp.dirname(template_path), 'avatar_initialize_actions_flickr30k_ent.txt')
        with open(output_path, 'w') as f:
            f.write(template)
            
        return output_path

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
        
        # Generate causal improvement guidelines
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
                
        # Combine sections with template
        prompt = template.replace('<feedback_message>', feedback)
        prompt = prompt.replace('<discovery_insights>', discovery_section)
        prompt = prompt.replace('<reasoning_guidelines>', reasoning_section)
        prompt = prompt.replace('<improvement_suggestions>', improvement_section)
        
        return prompt
        
    def _extract_causal_insights(self) -> Dict[str, Any]:
        """
        Extract key causal insights from discovery and analysis outputs
        
        Returns:
            Dictionary containing extracted causal insights with weights
        """
        insights = {}
        
        # Extract and process causal relationships with weights
        raw_relationships = self.discovery_output.get('causal_relationships', [])
        insights['causal_relationships'] = [
            {
                'cause': rel[0],
                'effect': rel[1],
                'strength': self.discovery_output['edge_weights'][i],
                'direct_effect': 0.0,
                'indirect_effect': 0.0
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
            
            # Adjust strength based on effect ratio
            effect_ratio = mediation_effects.get('effect_ratio', 0.5)
            rel['strength'] = rel['strength'] * (1 + effect_ratio * 0.2)
            
        # Extract key variables with their causal importance
        insights['key_variables'] = self._identify_key_variables()
        insights['mediation_effects'] = mediation_effects
        
        return insights
        
    def _generate_discovery_section(self, insights: Dict[str, Any]) -> str:
        """
        Generate discovery insights section with causal weights
        
        Args:
            insights: Extracted causal insights with weights
            
        Returns:
            Formatted discovery section string with causal weights
        """
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
                    f"indirect: {rel['indirect_effect']:.2f})\n"
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
                
        # Add key variables with causal importance
        if insights['key_variables']:
            section += "- Key Variables with Causal Importance:\n"
            for var in insights['key_variables']:
                # Calculate importance score based on connected relationships
                importance = sum(
                    rel['strength'] for rel in insights['causal_relationships']
                    if var in [rel['cause'], rel['effect']]
                )
                section += f"  * {var} (importance: {importance:.2f})\n"
                
        return section
        
    def _generate_reasoning_section(self, insights: Dict[str, Any]) -> str:
        """
        Generate reasoning guidelines with causal weights
        
        Args:
            insights: Extracted causal insights with weights
            
        Returns:
            Formatted reasoning section string with causal guidance
        """
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
            sorted_vars = sorted(
                insights['key_variables'],
                key=lambda var: sum(
                    rel['strength'] for rel in insights['causal_relationships']
                    if var in [rel['cause'], rel['effect']]
                ),
                reverse=True
            )
            for var in sorted_vars:
                importance = sum(
                    rel['strength'] for rel in insights['causal_relationships']
                    if var in [rel['cause'], rel['effect']]
                )
                section += f"  * {var} (importance: {importance:.2f})\n"
                
        return section
        
    def _identify_key_variables(self) -> List[str]:
        """
        Identify key variables from discovery output
        
        Returns:
            List of key variable names
        """
        # Implement logic to identify key variables
        # This could be based on effect size, centrality, etc.
        return []
