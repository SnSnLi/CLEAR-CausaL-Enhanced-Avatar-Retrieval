import os.path as osp
import torch
from typing import Any
from avatar.models.model import ModelForQA
from tqdm import tqdm


class VSS(ModelForQA):
    
    def __init__(self, 
                 kb, 
                 query_emb_dir: str, 
                 candidates_emb_dir: str, 
                 emb_model: str = 'openai/clip-vit-large-patch14/image'):
        """
        Vector Similarity Search

        Args:
            kb: Knowledge base.
            query_emb_dir (str): Directory to query embeddings.
            candidates_emb_dir (str): Directory to candidate embeddings.
            emb_model (str): Embedding model name.
        """
        super().__init__(kb)
        self.emb_model = emb_model
        self.query_emb_dir = query_emb_dir
        self.candidates_emb_dir = candidates_emb_dir

        candidate_emb_path = osp.join(candidates_emb_dir, 'candidate_emb_dict.pt')
        if osp.exists(candidate_emb_path):
            candidate_emb_dict = torch.load(candidate_emb_path)
            print(f'Loaded candidate_emb_dict from {candidate_emb_path}!')
    # 加载后立即过滤，只保留2500个ID的embeddings
            candidate_emb_dict = {k: v for k, v in candidate_emb_dict.items() if k in self.candidate_ids}
        else:
            print('Loading candidate embeddings...')
            candidate_emb_dict = {}
            for idx in tqdm(self.candidate_ids):
                candidate_emb_dict[idx] = torch.load(osp.join(candidates_emb_dir, f'{idx}.pt'))
            torch.save(candidate_emb_dict, candidate_emb_path)
            print(f'Saved candidate_emb_dict to {candidate_emb_path}!')


        # 打印候选嵌入字典和候选 ID 的长度
        print(f"Length of candidate_emb_dict: {len(candidate_emb_dict)}")
        print(f"Length of self.candidate_ids: {len(self.candidate_ids)}")

        assert len(candidate_emb_dict) == len(self.candidate_ids)
        candidate_embs = [candidate_emb_dict[idx].view(1, -1) for idx in self.candidate_ids]
        self.candidate_embs = torch.cat(candidate_embs, dim=0)

    def forward(self, 
                query: str, 
                query_id: int, 
                **kwargs: Any) -> dict:
        """
        Forward pass to compute similarity scores for the given query.

        Args:
            query (str): Query string.
            query_id (int): Query index.

        Returns:
            pred_dict (dict): A dictionary of candidate ids and their corresponding similarity scores.
        """
        query_emb = self.get_query_emb(query, query_id, emb_model=self.emb_model)
        similarity = torch.matmul(query_emb.cuda(), self.candidate_embs.cuda().T).cpu().view(-1)
        pred_dict = {self.candidate_ids[i]: similarity[i] for i in range(len(self.candidate_ids))}
        return pred_dict
