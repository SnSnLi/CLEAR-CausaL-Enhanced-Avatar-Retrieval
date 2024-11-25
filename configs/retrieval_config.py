from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    # Model configuration
    hidden_size: int = 768
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Causal learning configuration
    use_causal: bool = True
    causal_lambda: float = 0.1
    dag_lambda: float = 0.1
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 1000
    
    # Dataset configuration
    dataset: str = "flickr30k_entities"
    emb_model: str = "openai/clip-vit-large-patch14"
    root_dir: str = "/root/onethingai-tmp/avatar""
    
    # LLM configuration
    agent_llm: str = "gpt-4o"
    api_func_llm: str = "gpt-4o"
    
    # Evaluation configuration
    eval_steps: int = 100
    eval_batch_size: int = 64
    topk_eval: int = 100
    topk_test: int = 10

    def __post_init__(self):
        if self.dataset == "prime":
            self.emb_model = "text-embedding-ada-002"
            self.agent_llm = "gpt-4o-mini"
            self.api_func_llm = "gpt-4o-mini"