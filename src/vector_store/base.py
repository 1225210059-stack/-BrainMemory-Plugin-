from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class BaseVectorStore(ABC):
    """向量存储抽象基类，支持Chroma/Milvus一键切换"""
    @abstractmethod
    def __init__(self, collection_name: str, embeddings: Embeddings, **kwargs):
        pass

    @abstractmethod
    def add_documents(self, docs: List[Document], batch_size: int = 100, async_insert: bool = False) -> List[str]:
        """批量插入文档，支持异步插入优化内存"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """向量相似度检索"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """删除指定文档"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查存储连接状态"""
        pass

    @abstractmethod
    def close(self):
        """关闭连接"""
        pass
