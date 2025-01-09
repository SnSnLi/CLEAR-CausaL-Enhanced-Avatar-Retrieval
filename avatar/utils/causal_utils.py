from typing import Dict, Any, List

def causal_mediation_analysis(discovery_output: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Perform causal mediation analysis on discovery output
    
    Args:
        discovery_output: Dictionary containing causal relationships and variables
        
    Returns:
        List of mediation effects, each containing:
        - cause: The direct cause variable
        - effect: The final effect variable
        - mediator: The mediating variable
    """
    mediation_effects = []
    
    # Extract causal relationships
    relationships = discovery_output.get('causal_relationships', [])
    
    # Find mediation effects by looking for chains of relationships
    for rel1 in relationships:
        for rel2 in relationships:
            # Check if rel1's effect is rel2's cause
            if rel1['effect'] == rel2['cause']:
                mediation_effects.append({
                    'cause': rel1['cause'],
                    'mediator': rel1['effect'],
                    'effect': rel2['effect']
                })
    
    return mediation_effects

def calculate_mediation_strength(mediation_effects: List[Dict[str, str]], 
                                discovery_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate strength of mediation effects based on relationship strengths
    
    Args:
        mediation_effects: List of mediation effects from causal_mediation_analysis
        discovery_output: Original discovery output containing relationship strengths
        
    Returns:
        List of mediation effects with strength calculations
    """
    # Create lookup for relationship strengths
    strength_lookup = {
        (rel['cause'], rel['effect']): rel['strength']
        for rel in discovery_output.get('causal_relationships', [])
    }
    
    # Calculate mediation strength as product of individual relationship strengths
    for effect in mediation_effects:
        cause_mediator = (effect['cause'], effect['mediator'])
        mediator_effect = (effect['mediator'], effect['effect'])
        
        strength1 = strength_lookup.get(cause_mediator, 0)
        strength2 = strength_lookup.get(mediator_effect, 0)
        
        effect['strength'] = strength1 * strength2
    
    return mediation_effects
