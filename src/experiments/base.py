# src/your_proj/experiments/base.py
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple
from transformers import PreTrainedTokenizerBase, PreTrainedModel

class Experiment(ABC):
    @abstractmethod
    def load_model_and_tok(self, cfg) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]: ...
    @abstractmethod
    def load_datasets(self) -> Tuple[Any, Any]: ...
    @abstractmethod
    def preprocessing_fn(self, tok: PreTrainedTokenizerBase, model: PreTrainedModel) -> Callable: ...
    @abstractmethod
    def get_collator(self, tok: PreTrainedTokenizerBase) -> Callable: ...
    @abstractmethod
    def get_trainer_class(self) -> Any: ...
    
from .bias import BIAS
from .reasoning import REASONING
from .reranker import RERANKER
from .safety import SAFETY
from .dpo_kl import DPO_KL

EXPERIMENTS = {
    "bias": BIAS,
    "reasoning": REASONING,
    "reranker": RERANKER,
    "safety": SAFETY,
    "dpo_kl": DPO_KL,
}