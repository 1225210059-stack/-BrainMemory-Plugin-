import os
import json
from datetime import datetime
from typing import Dict
from abc import ABC, abstractmethod
import boto3
from botocore.exceptions import ClientError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from .encryption import Encryptor

class BaseStorage(ABC):
    """存储抽象基类，所有存储实现必须继承该类"""
    @abstractmethod
    def save_metadata(self, metadata: Dict) -> None:
        """保存记忆元数据"""
        pass

    @abstractmethod
    def load_metadata(self) -> Dict:
        """加载记忆元数据"""
        pass

    @abstractmethod
    def save_vector_doc(self, doc: Document) -> None:
        """保存向量文档"""
        pass

    @abstractmethod
    def get_vector_db(self) -> Chroma:
        """获取向量数据库实例"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查存储连接状态"""
        return True

class LocalStorage(BaseStorage):
    """纯本地存储实现"""
    def __init__(self, embeddings: Embeddings, local_path: str = "./local_memory_db"):
        self.local_path = local_path
        self.metadata_path = os.path.join(local_path, "memory_metadata.json")
        self.embeddings = embeddings
        os.makedirs(local_path, exist_ok=True)
        os.makedirs(os.path.join(local_path, "vector_db"), exist_ok=True)
        
        self.vector_db = Chroma(
            collection_name="brain_memory_local",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(local_path, "vector_db")
        )

    def save_metadata(self, metadata: Dict) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    def load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"short_term_memory": [], "long_term_memory": [], "updated_at": datetime.now().isoformat()}

    def save_vector_doc(self, doc: Document) -> None:
        self.vector_db.add_documents([doc])

    def get_vector_db(self) -> Chroma:
        return self.vector_db

    def is_connected(self) -> bool:
        return os.path.exists(self.local_path)

class CloudStorage(BaseStorage):
    """纯云端存储实现，兼容所有S3协议存储，端到端加密"""
    def __init__(
        self,
        embeddings: Embeddings,
        access_key: str,
        secret_key: str,
        endpoint_url: str,
        region: str,
        bucket_name: str,
        encryptor: Encryptor,
        cache_path: str = "./cloud_cache"
    ):
        self.encryptor = encryptor
        self.bucket_name = bucket_name
        self.metadata_key = "memory_metadata.json.encrypted"
        self.vector_prefix = "vector_db/"
        self.cache_path = cache_path
        
        # 初始化S3客户端
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name=region
        )
        
        # 本地缓存兜底，断网不影响使用
        self.local_cache = LocalStorage(embeddings, cache_path)
        self._check_bucket()

    def _check_bucket(self) -> None:
        """检查存储桶，不存在则创建"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.s3_client.create_bucket(Bucket=self.bucket_name)

    def is_connected(self) -> bool:
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception:
            return False

    def save_metadata(self, metadata: Dict) -> None:
        # 先更新本地缓存
        self.local_cache.save_metadata(metadata)
        # 云端连接正常则同步加密上传
        if self.is_connected():
            encrypted_data = self.encryptor.encrypt(metadata)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.metadata_key,
                Body=encrypted_data
            )

    def load_metadata(self) -> Dict:
        # 优先从云端加载
        if self.is_connected():
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.metadata_key)
                encrypted_data = response["Body"].read()
                metadata = self.encryptor.decrypt(encrypted_data)
                self.local_cache.save_metadata(metadata)
                return metadata
            except Exception:
                return self.local_cache.load_metadata()
        # 断网则使用本地缓存
        return self.local_cache.load_metadata()

    def save_vector_doc(self, doc: Document) -> None:
        self.local_cache.save_vector_doc(doc)
        if self.is_connected():
            doc_key = f"{self.vector_prefix}{doc.metadata['chunk_id']}.json.encrypted"
            doc_data = {"page_content": doc.page_content, "metadata": doc.metadata}
            encrypted_data = self.encryptor.encrypt(doc_data)
            self.s3_client.put_object(Bucket=self.bucket_name, Key=doc_key, Body=encrypted_data)

    def get_vector_db(self) -> Chroma:
        return self.local_cache.get_vector_db()

class HybridStorage(BaseStorage):
    """混合存储实现：重要内容本地存储，普通内容云端同步"""
    def __init__(self, local_storage: LocalStorage, cloud_storage: CloudStorage):
        self.local = local_storage
        self.cloud = cloud_storage

    def save_metadata(self, metadata: Dict) -> None:
        # 全量数据本地存储
        self.local.save_metadata(metadata)
        # 过滤重要内容，仅同步普通内容到云端
        cloud_metadata = metadata.copy()
        cloud_metadata["long_term_memory"] = [
            m for m in metadata["long_term_memory"] if not m.get("is_important", False)
        ]
        self.cloud.save_metadata(cloud_metadata)

    def load_metadata(self) -> Dict:
        # 合并本地+云端数据，本地优先
        local_meta = self.local.load_metadata()
        cloud_meta = self.cloud.load_metadata()
        
        local_chunk_ids = [m["chunk_id"] for m in local_meta["long_term_memory"]]
        for m in cloud_meta["long_term_memory"]:
            if m["chunk_id"] not in local_chunk_ids:
                local_meta["long_term_memory"].append(m)
        return local_meta

    def save_vector_doc(self, doc: Document) -> None:
        is_important = doc.metadata.get("is_important", False)
        self.local.save_vector_doc(doc)
        if not is_important:
            self.cloud.save_vector_doc(doc)

    def get_vector_db(self) -> Chroma:
        return self.local.get_vector_db()

    def is_connected(self) -> bool:
        return self.local.is_connected()
