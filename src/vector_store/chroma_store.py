from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from .base import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    """原有Chroma向量存储，向下兼容"""
    def __init__(self, collection_name: str, embeddings: Embeddings, persist_directory: str = "./local_memory_db", **kwargs):
        self._db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    def add_documents(self, docs: List[Document], batch_size: int = 100, async_insert: bool = False) -> List[str]:
        return self._db.add_documents(docs)

    def similarity_search(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        return self._db.similarity_search(query, k=top_k, **kwargs)

    def delete(self, ids: List[str]) -> bool:
        self._db.delete(ids=ids)
        return True

    def is_connected(self) -> bool:
        return self._db._client is not None

    def close(self):
        pass
