from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CausalConfig:
    # 模型结构配置
    hidden_size: int = 768 
    unified_dim: int = 768
    num_attention_heads: int = 12
    num_causal_layers: int = 2
    dropout_rate: float = 0.1
    
    # 训练相关配置
    causal_loss_weight: float = 1.0
    invariance_weight: float = 0.1
    use_causal_mask: bool = True
    
    # 优化器配置
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # 训练控制
    max_epochs: int = 100
    eval_steps: int = 100
    save_steps: int = 1000

    # 因果推理相关
    use_causal_attention: bool = True
    causal_dropout: float = 0.1
    use_invariance_loss: bool = True

    def __post_init__(self):
        # 确保维度匹配
        assert self.hidden_size == self.unified_dim, "hidden_size must match unified_dim"