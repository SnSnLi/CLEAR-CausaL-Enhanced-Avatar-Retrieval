import os
import os.path as osp
from typing import Any, Union, List, Dict

import torch
import torch.nn as nn
from stark_qa.tools.api import get_openai_embedding
from stark_qa.evaluator import Evaluator
from avatar.tools import GetCLIPTextEmbedding
from transformers import AutoModel
from .feature_extractor import FeatureExtractor
from .causal_graph_learning import CausalGraphLearning
from .causal_attention import CausalAttention
from .retrieval_model import RetrievalModel
from .evaluator import Evaluator
from pathlib import Path as osp


class ModelForQA(nn.Module):
    
    def __init__(self, kb):
        """
        Initializes the model with the given knowledge base.
        
        Args:
            kb: Knowledge base containing candidate information.
        """
        super(ModelForQA, self).__init__()
        self.kb = kb

        self.candidate_ids = kb.candidate_ids
        self.num_candidates = kb.num_candidates
        self.query_emb_dict = {}
        self.evaluator = Evaluator(self.candidate_ids)
        # 原有的特征提取器和编码器
        self.feature_extractor = FeatureExtractor(config)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)
        
        # 新增：因果图学习模块
        self.causal_graph_learner = CausalGraphLearning(
            feature_dim=config.hidden_size,
            hidden_dim=config.hidden_size
        )
        
        # 新增：因果注意力模块
        self.causal_attention = CausalAttention(
            hidden_dim=config.hidden_size
        )
        
        # 新增：检索模型
        self.retrieval_model = RetrievalModel(
            feature_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            output_dim=config.hidden_size
        )


    
    
    def forward(self, 
                query: Union[str, List[str]], 
                candidates: List[int] = None,
                query_id: Union[int, List[int]] = None,
                **kwargs: Any) -> Dict[str, Any]:
        """
        Forward pass to compute predictions for the given query.
        
        Args:
            query (Union[str, list]): Query string or a list of query strings.
            candidates (Union[list, None]): A list of candidate ids (optional).
            query_id (Union[int, list, None]): Query index (optional).
            
        Returns:
            pred_dict (dict): A dictionary of predicted scores or answer ids.
        """
        if isinstance(query, str):
            query = [query]
        if isinstance(query_id, int):
            query_id = [query_id]
        
        # 获取查询嵌入
        query_embs = []
        for q, q_id in zip(query, query_id):
            query_emb = self.get_query_emb(q, q_id)
            query_embs.append(query_emb)
        query_embs = torch.stack(query_embs)
        
        # 提取特征
        visual_features = self.feature_extractor(kwargs['images'])
        text_features = query_embs
        
        # 因果增强的检索
        retrieval_outputs = self.retrieval_model(
            images=visual_features['global_features'],
            texts=text_features
        )
        
        # 整理输出
        outputs = {
            'visual_features': visual_features,
            'text_features': text_features,
            'matching_scores': retrieval_outputs['matching_scores'],
            'causal_info': retrieval_outputs['causal_info'],
            'img_features': retrieval_outputs['img_features'],
            'text_features': retrieval_outputs['text_features']
        }
        
        # 计算损失（如果在训练阶段）
        if self.training and 'labels' in kwargs:
            retrieval_loss = self.retrieval_model.compute_loss(
                retrieval_outputs['matching_scores'],
                kwargs['labels']
            )
            outputs['loss'] = retrieval_loss
        
        return outputs
    
    def get_query_emb(self, 
                       query: str, 
                       query_id: int, 
                       emb_model: str = 'text-embedding-ada-002') -> torch.Tensor:
        """
        Retrieves or computes the embedding for the given query.
        
        Args:
            query (str): Query string.
            query_id (int): Query index.
            emb_model (str): Embedding model to use.
            
        Returns:
            query_emb (torch.Tensor): Query embedding.
        """
        if query_id is None:
            query_emb = get_openai_embedding(query, model=self.emb_model)
        elif len(self.query_emb_dict) > 0:
            query_emb = self.query_emb_dict[query_id]
        else:
            query_emb_dic_path = osp.join(self.query_emb_dir, 'query_emb_dict.pt')
            if os.path.exists(query_emb_dic_path):
                print(f'Load query embeddings from {query_emb_dic_path}')
                self.query_emb_dict = torch.load(query_emb_dic_path)
                query_emb = self.query_emb_dict[query_id]
            else:
                query_emb_dir = osp.join(self.query_emb_dir, 'query_embs')
                if not os.path.exists(query_emb_dir):
                    os.makedirs(query_emb_dir)
                query_emb_path = osp.join(query_emb_dir, f'query_{query_id}.pt')
                query_emb = get_openai_embedding(query, model=self.emb_model)
                torch.save(query_emb, query_emb_path)
        return query_emb

    def compute_similarity(self, visual_features, text_features):
        """使用因果增强的匹配分数替代简单的相似度计算"""
        retrieval_outputs = self.retrieval_model(
            images=visual_features,
            texts=text_features
        )
        return retrieval_outputs['matching_scores']
    
    def evaluate(self, 
                 pred_dict: Dict[int, float], 
                 answer_ids: torch.LongTensor, 
                 metrics: List[str] = ['mrr', 'hit@3', 'recall@20'], 
                 **kwargs: Any) -> Dict[str, float]:
        """
        Evaluates the predictions using the specified metrics.
        
        Args:
            pred_dict (Dict[int, float]): Predicted answer ids or scores.
            answer_ids (torch.LongTensor): Ground truth answer ids.
            metrics (List[str]): A list of metrics to be evaluated, including 'mrr', 'hit@k', 'recall@k', 
                                 'precision@k', 'map@k', 'ndcg@k'.
                             
        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        return self.evaluator(pred_dict, answer_ids, metrics)

def build_avatar_model(config):
    # 添加因果检索相关的配置
    if not hasattr(config, 'causal_hidden_dim'):
        config.causal_hidden_dim = config.hidden_size
    if not hasattr(config, 'causal_output_dim'):
        config.causal_output_dim = config.hidden_size
    
    model = ModelForQA(kb=config.kb, config=config)
    
    # 加载预训练权重
    if config.pretrained_path:
        state_dict = torch.load(
            config.pretrained_path,
            map_location='cpu'
        )
        model.load_state_dict(state_dict)
    
    return model
