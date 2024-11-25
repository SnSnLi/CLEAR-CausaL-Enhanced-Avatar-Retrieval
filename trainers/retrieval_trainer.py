import os
import torch
import logging
from tqdm import tqdm
from typing import Dict, Optional

from avatar.models import AvatarModel
from avatar.datasets import RetrievalDataset
from avatar.utils.metrics import compute_metrics
from configs.retrieval_config import RetrievalConfig

class RetrievalTrainer:
    def __init__(
        self,
        model: AvatarModel,
        config: RetrievalConfig,
        train_dataset: RetrievalDataset,
        eval_dataset: Optional[RetrievalDataset] = None,
    ):
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
            progress_bar = tqdm(range(len(self.train_dataset) // self.config.batch_size))
            
            for step in progress_bar:
                batch = self.train_dataset.get_batch(self.config.batch_size)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_steps += 1
                
                progress_bar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                
                if total_steps % self.config.eval_steps == 0:
                    metrics = self.evaluate()
                    self.model.train()
                    
                    # Save best model
                    if metrics['mrr'] > best_metric:
                        best_metric = metrics['mrr']
                        self.save_model()
                        
            avg_loss = epoch_loss / (step + 1)
            logging.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
            
    def evaluate(self):
        self.model.eval()
        if self.eval_dataset is None:
            return {}
            
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(self.eval_dataset), self.config.eval_batch_size):
                batch = self.eval_dataset.get_batch(self.config.eval_batch_size)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
        metrics = compute_metrics(all_predictions, all_labels)
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics
        
    def save_model(self, path: Optional[str] = None):
        if path is None:
            path = f"checkpoints/{self.config.dataset}_model.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))