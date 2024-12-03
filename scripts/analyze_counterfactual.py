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
    parser.add_argument("--output_dir", type=str, default="outputs/counterfactual")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=100)
    return parser.parse_args()

def analyze_counterfactuals(model, test_data, config, n_samples):
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Analyzing"):
            features, adj_matrix = batch
            features = features.to(args.device)
            adj_matrix = adj_matrix.to(args.device)
            
            # 对每个样本进行反事实分析
            for i in range(features.size(0)):
                # 生成反事实特征
                cf_value = features[i] + torch.randn_like(features[i]) * config.intervention_strength
                
                # 计算反事实效应
                effect = compute_counterfactual_effect(
                    adj_matrix,
                    features,
                    source=i,
                    target=-1,
                    intervention_value=cf_value
                )
                
                # 进行因果归因分析
                attributions = causal_attribution(
                    adj_matrix,
                    features,
                    target_idx=i,
                    n_samples=n_samples
                )
                
                results.append({
                    "sample_id": i,
                    "counterfactual_effect": effect,
                    "attributions": attributions
                })
    
    return results

def main():
    args = parse_args()
    config = RetrievalConfig()
    
    # 加载模型
    model = CausalModel(
        hidden_size=config.causal_hidden_dim,
        output_size=config.causal_output_dim
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    
    # 加载测试数据
    test_data = load_flickr30k_data(split="test")
    
    # 进行反事实分析
    results = analyze_counterfactuals(
        model, 
        test_data, 
        config,
        args.n_samples
    )
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "counterfactual_analysis.txt")
    
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Sample {result['sample_id']}:\n")
            f.write(f"Counterfactual Effect: {result['counterfactual_effect']:.4f}\n")
            f.write("Causal Attributions:\n")
            for node, attribution in result['attributions'].items():
                f.write(f"  Node {node}: {attribution:.4f}\n")
            f.write("\n")
    
    logging.info(f"Counterfactual analysis results saved to {output_file}")

if __name__ == "__main__":
    main()