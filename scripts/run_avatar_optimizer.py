import sys
sys.path.append('.')

import os
import os.path as osp
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

import torch
import stark_qa
from avatar.models import get_model
from stark_qa.tools.seed import set_seed
from avatar.kb import Flickr30kEntities
from avatar.qa_datasets import QADataset
from scripts.args import parse_args_w_defaults
from models.causal_model import CausalModel
from utils.causal_utils import causal_mediation_analysis, compute_counterfactual_effect
from causal_trainer import CausalTrainer
from evaluate_causal import evaluate_causal_effects
from analyze_counterfactual import analyze_counterfactuals

if __name__ == '__main__':
    try:
        args = parse_args_w_defaults('config/default_args.json')
        set_seed(args.seed)
        
        # 添加因果学习相关参数
        if not hasattr(args, 'use_causal'):
            args.use_causal = True  # 默认启用因果学习
        if not hasattr(args, 'causal_hidden_dim'):
            args.causal_hidden_dim = 256
        if not hasattr(args, 'causal_output_dim'):
            args.causal_output_dim = 128
        if not hasattr(args, 'causal_weight'):
            args.causal_weight = 0.1
        if not hasattr(args, 'invariance_weight'):
            args.invariance_weight = 0.1
        
        # 设置数据目录
        if args.dataset in ['amazon', 'mag', 'prime']:
            emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
            args.query_emb_dir = osp.join(emb_root, 'query')
            args.node_emb_dir = osp.join(emb_root, 'doc')
            args.chunk_emb_dir = osp.join(emb_root, 'chunk')
            os.makedirs(args.query_emb_dir, exist_ok=True)
            os.makedirs(args.node_emb_dir, exist_ok=True)
            os.makedirs(args.chunk_emb_dir, exist_ok=True)

            kb = stark_qa.load_skb(args.dataset)
            qa_dataset = stark_qa.load_qa(args.dataset)

        elif args.dataset == 'flickr30k_entities':
            emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
            args.chunk_emb_dir = None
            args.query_emb_dir = osp.join(emb_root, 'query')
            args.node_emb_dir = osp.join(emb_root, 'image')
            os.makedirs(args.query_emb_dir, exist_ok=True)
            os.makedirs(args.node_emb_dir, exist_ok=True)

            kb = Flickr30kEntities(root=args.root_dir)
            qa_dataset = QADataset(name=args.dataset, root=args.root_dir)
        
        # 初始化模型
        model = get_model(args, kb)
        model.parent_pred_path = osp.join(args.output_dir, f'eval/{args.dataset}/VSS/{args.emb_model}/eval_results_test.csv')
        
        # 初始化因果模型和训练器
        if args.use_causal:
            causal_model = CausalModel(
                hidden_size=args.causal_hidden_dim,
                output_size=args.causal_output_dim
            ).to(model.device)
            model.causal_model = causal_model
            
            trainer = CausalTrainer(
                model=model,
                config=args,
                train_dataset=qa_dataset.get_split('train'),
                eval_dataset=qa_dataset.get_split('val')
            )
        
        # 生成训练分组
        if args.dataset in ['amazon', 'mag', 'prime']:
            group = model.generate_group(qa_dataset, batch_size=5, n_init_examples=200, split='train')
            group = model.generate_group(qa_dataset, batch_size=5, split='val')
            group = model.generate_group(qa_dataset, batch_size=5, split='test')

        # 定义评估指标
        metrics = [
            'mrr', 'map', 'rprecision',
            'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100',
            'hit@1', 'hit@3', 'hit@5', 'hit@10', 'hit@20', 'hit@50'
        ]
        
        # 添加因果相关的评估指标
        if args.use_causal:
            metrics.extend(['causal_acc', 'causal_loss', 'invariance_score'])
            
        # 训练和优化
        if args.use_causal:
            # 使用因果训练器进行训练
            trainer.train()
            
            # 进行因果评估
            evaluate_causal_effects(
                model=model,
                dataset=qa_dataset.get_split('test'),
                output_dir=args.output_dir
            )
            
            # 进行反事实分析
            analyze_counterfactuals(
                model=model,
                dataset=qa_dataset.get_split('test'),
                output_dir=args.output_dir
            )
        
        # 常规优化过程
        model.optimize_actions(
            qa_dataset=qa_dataset, 
            seed=args.seed, 
            group_idx=args.group_idx, 
            use_group=args.use_group,
            n_eval=args.n_eval,
            n_examples=args.n_examples,
            n_total_steps=args.n_total_steps,
            topk_eval=args.topk_eval,
            topk_test=args.topk_test,
            batch_size=args.batch_size,
            metrics=metrics
        )
        
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        import traceback
        traceback.print_exc()
        time.sleep(60)  