import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict
from pymilvus import MilvusClient, DataType, Collection, connections, utility
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from .base import BaseVectorStore

class MilvusVectorStore(BaseVectorStore):
    """Milvus高性能向量存储，支持增量检索、批量异步插入"""
    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        uri: str = "http://localhost:19530",
        token: str = "",
        dim: int = 384,
        **kwargs
    ):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.dim = dim
        self.uri = uri
        self.token = token

        # 初始化Milvus客户端
        self.client = MilvusClient(uri=uri, token=token)
        self._init_collection()
        self._create_index()

        # 异步批量插入队列
        self._insert_queue = asyncio.Queue()
        self._batch_task = None

    def _init_collection(self):
        """初始化集合，兼容增量插入"""
        if self.client.has_collection(self.collection_name):
            return
        
        # 集合schema设计，兼容原方案的字段
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            description="BrainMemory-Plugin 类脑记忆向量库"
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="tags", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="is_important", datatype=DataType.BOOL)
        schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=65535)

        # 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Session"
        )

    def _create_index(self):
        """创建IVF_FLAT索引，检索速度提升10-100倍"""
        if self.client.has_index(self.collection_name, index_name="embedding_idx"):
            return
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 1024}
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        # 加载集合到内存
        self.client.load_collection(self.collection_name)

    def add_documents(self, docs: List[Document], batch_size: int = 100, async_insert: bool = False) -> List[str]:
        """批量插入文档，支持异步插入降低内存峰值"""
        if not docs:
            return []
        
        # 拆分批次，优化内存占用
        batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
        all_ids = []

        for batch in batches:
            # 生成嵌入向量
            texts = [doc.page_content for doc in batch]
            embeddings = self.embeddings.embed_documents(texts)
            
            # 构建插入数据
            insert_data = []
            for idx, doc in enumerate(batch):
                chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("id", f"chunk_{datetime.now().timestamp()}"))
                insert_data.append({
                    "chunk_id": chunk_id,
                    "embedding": embeddings[idx],
                    "timestamp": doc.metadata.get("created_at", datetime.now().isoformat()),
                    "tags": doc.metadata.get("tags", ""),
                    "is_important": doc.metadata.get("is_important", False),
                    "page_content": doc.page_content
                })
                all_ids.append(chunk_id)
            
            # 同步插入
            if not async_insert:
                self.client.insert(collection_name=self.collection_name, data=insert_data)
            else:
                # 异步插入（非阻塞）
                asyncio.create_task(self._async_insert_batch(insert_data))

        return all_ids

    async def _async_insert_batch(self, batch_data: List[Dict]):
        """异步批量插入，减少主线程内存峰值"""
        try:
            self.client.insert(collection_name=self.collection_name, data=batch_data)
        except Exception as e:
            print(f"[Milvus] 异步插入失败: {str(e)}")

    def similarity_search(self, query: str, top_k: int = 5, filter_expr: str = None, **kwargs) -> List[Document]:
        """增量检索，支持标签过滤、相似度排序"""
        # 生成查询向量
        query_embedding = self.embeddings.embed_query(query)
        
        # 检索参数优化，nprobe动态调整
        nprobe = min(64, top_k * 2)
        search_params = {"metric_type": "IP", "params": {"nprobe": nprobe}}
        
        # 执行检索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=top_k,
            filter=filter_expr,
            output_fields=["chunk_id", "timestamp", "tags", "is_important", "page_content"]
        )

        # 格式化结果
        docs = []
        for hit in results[0]:
            doc = Document(
                page_content=hit.get("page_content", ""),
                metadata={
                    "chunk_id": hit.get("chunk_id"),
                    "timestamp": hit.get("timestamp"),
                    "tags": hit.get("tags", "").split(","),
                    "is_important": hit.get("is_important"),
                    "similarity": hit.score
                }
            )
            docs.append(doc)
        return docs

    def delete(self, ids: List[str]) -> bool:
        """删除指定记忆块"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"chunk_id in [{','.join([f'\'{id}\'' for id in ids])}]"
            )
            return True
        except Exception as e:
            print(f"[Milvus] 删除失败: {str(e)}")
            return False

    def is_connected(self) -> bool:
        """检查连接状态"""
        try:
            return self.client.has_collection(self.collection_name)
        except Exception:
            return False

    def close(self):
        """关闭连接"""
        self.client.close()

    def optimize_index(self):
        """索引优化，定期执行提升检索性能"""
        self.client.release_collection(self.collection_name)
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=self.client.prepare_index_params().add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 1024}
            ),
            sync=True
        )
        self.client.load_collection(self.collection_name)
        print("[Milvus] 索引优化完成")
