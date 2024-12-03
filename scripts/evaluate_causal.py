import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from utils.causal_utils import *
from utils.data_utils import load_flickr30k_data
from models.causal_model import CausalModel
from retrieval_config import RetrievalConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def causal_mediation_analysis(adj_matrix, features, treatment, mediator, outcome):
    """计算因果中介效应"""
    direct_effect = compute_direct_effect(adj_matrix, features, treatment, outcome)
    indirect_effect = compute_indirect_effect(adj_matrix, features, treatment, mediator, outcome)
    total_effect = direct_effect + indirect_effect
    
    return {
        "natural_direct_effect": direct_effect,
        "natural_indirect_effect": indirect_effect,
        "total_effect": total_effect
    }

def compute_direct_effect(adj_matrix, features, treatment, outcome):
    treatment_effect = (adj_matrix[:, treatment] * features[:, outcome]).mean()
    return float(treatment_effect)

def compute_indirect_effect(adj_matrix, features, treatment, mediator, outcome):
    path1 = adj_matrix[:, treatment] * features[:, mediator]
    path2 = adj_matrix[:, mediator] * features[:, outcome]
    indirect_effect = (path1 * path2).mean()
    return float(indirect_effect)

def evaluate_causal_effects(model, test_data, config):
    results = {
        "direct_effects": [],
        "indirect_effects": [],
        "total_effects": []
    }
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Evaluating"):
            features, adj_matrix = batch
            features = features.to(args.device)
            adj_matrix = adj_matrix.to(args.device)
            
            for i in range(features.size(0)):
                effects = causal_mediation_analysis(
                    adj_matrix,
                    features,
                    treatment=i,
                    mediator=i+1 if i < features.size(0)-1 else 0,
                    outcome=-1
                )
                
                results["direct_effects"].append(effects["natural_direct_effect"])
                results["indirect_effects"].append(effects["natural_indirect_effect"]) 
                results["total_effects"].append(effects["total_effect"])
    
    return results

def main():
    args = parse_args()
    config = RetrievalConfig()
    
    model = CausalModel(
        hidden_size=config.causal_hidden_dim,
        output_size=config.causal_output_dim
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    
    test_data = load_flickr30k_data(split="test")
    results = evaluate_causal_effects(model, test_data, config)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "causal_effects.txt")
    
    with open(output_file, "w") as f:
        for key, values in results.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            f.write(f"{key}: Mean = {mean_value:.4f}, Std = {std_value:.4f}\n")
    
    logging.info(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    main()