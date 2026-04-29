from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLM(ABC):
    """LLM抽象基类，支持OpenAI/LLaMA-3一键切换"""
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def chat(self, user_input: str, context: str = "", chat_history: List[Dict] = None) -> str:
        """带上下文的对话接口"""
        pass

    @abstractmethod
    def encode(self, text: str) -> str:
        """文本编码，用于精细编码"""
        pass
