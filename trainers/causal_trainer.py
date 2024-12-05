import os
import torch
import logging
from tqdm import tqdm
from typing import Optional

from avatar.models import AvaTaR
from avatar.kb.flickr30k_entities import Flickr30kEntities
from avatar.utils.metrics import compute_metrics
from configs.retrieval_config import RetrievalConfig

class CausalTrainer:
    def __init__(
        self,
        model: AvaTaR,
        config: RetrievalConfig,
        train_dataset: Flickr30kEntities,
        eval_dataset: Optional[Flickr30kEntities] = None,
    ):
        # 保持原有初始化
        super().__init__()
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        self.model.train()
        total_steps = 0
        best_metric = float('-inf')
        
        for epoch in range(self.config.max_epochs):
            epoch_loss = 0
            progress_bar = tqdm(enumerate(self.train_dataset), total=len(self.train_dataset))
            
            for step, batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 增加因果学习步骤
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                
                # 计算总损失 = AVATAR损失 + 因果损失
                causal_outputs = outputs['causal_outputs']
                total_loss = outputs.loss + causal_outputs['invariance_loss'] + beta * causal_outputs['sparsity_loss']  
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                total_steps += 1
                
                progress_bar.set_description(f"Epoch {epoch} Loss: {total_loss.item():.4f}")
                
                if total_steps % self.config.eval_steps == 0:
                    metrics = self.evaluate()
                    self.model.train()
                    
                    # 同时考虑检索和因果指标
                    combined_metric = metrics['retrieval_score'] + metrics['causal_score'] 
                    if combined_metric > best_metric:
                        best_metric = combined_metric
                        self.save_model()
                        
            avg_loss = epoch_loss / (step + 1)
            logging.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    def evaluate(self):
        # 保持原有评估逻辑
        self.model.eval()
        if self.eval_dataset is None:
            return {}
            
        metrics = {}
        with torch.no_grad():
            for batch in self.eval_dataset:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                # 增加因果评估指标
                causal_outputs = outputs['causal_outputs']
                metrics['causal_score'] = causal_outputs['causal_metric']
                metrics.update(outputs.metrics)
        
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics

    # save_model和load_model保持不变