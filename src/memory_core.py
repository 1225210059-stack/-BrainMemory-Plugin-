import os
import json
from datetime import datetime, timedelta
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from .vector_store import BaseVectorStore, ChromaVectorStore, MilvusVectorStore
from .summarizer import DynamicSummarizer
from .security import SecuritySanitizer
from .hybrid_model import BaseLLM, OpenAILLM, Llama3LLM

load_dotenv()

class MemoryChunk(BaseModel):
    """记忆块核心模型，兼容原有结构，支持增量优化"""
    chunk_id: str = Field(default_factory=lambda: os.urandom(8).hex())
    content: str
    structured_summary: str
    related_knowledge: List[str] = Field(default_factory=list)
    memory_strength: float = Field(default=1.0, ge=0, le=10)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    memory_level: str = Field(description="working/short_term/long_term", default="short_term")
    is_important: bool = False
    version: int = 1
    topic: str = "default"

class OptimizedBrainMemory:
    """优化后的类脑记忆核心引擎，整合所有优化模块"""
    def __init__(
        self,
        llm: BaseLLM = None,
        embeddings: Embeddings = None,
        vector_store: BaseVectorStore = None,
        storage_mode: str = "local",
        **kwargs
    ):
        # 加载核心配置
        self.WORKING_MEMORY_MAX_TOKEN = int(kwargs.get("working_memory_max_token", os.getenv("WORKING_MEMORY_MAX_TOKEN", 4000)))
        self.SHORT_TERM_MEMORY_DAYS = int(kwargs.get("short_term_memory_days", os.getenv("SHORT_TERM_MEMORY_DAYS", 14)))
        self.MEMORY_DECAY_RATE = float(kwargs.get("memory_decay_rate", os.getenv("MEMORY_DECAY_RATE", 0.1)))
        self.RETRIEVAL_STRENGTH_BOOST = float(kwargs.get("retrieval_strength_boost", os.getenv("RETRIEVAL_STRENGTH_BOOST", 0.8)))
        self.LOCAL_MEMORY_PATH = kwargs.get("local_path", os.getenv("LOCAL_MEMORY_PATH", "./local_memory_db"))

        # 初始化LLM（默认OpenAI兼容）
        self.llm = llm if llm else OpenAILLM(
            model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "")
        )

        # 初始化嵌入模型
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "")
        )

        # 初始化向量存储（Milvus/Chroma自动切换）
        self.vector_store = vector_store if vector_store else self._init_vector_store(storage_mode, **kwargs)

        # 初始化优化模块
        self.summarizer = DynamicSummarizer(device=kwargs.get("device", "auto"))
        self.sanitizer = SecuritySanitizer(custom_sensitive_keywords=kwargs.get("custom_sensitive_keywords"))

        # 三级记忆分层
        self.working_memory: List[MemoryChunk] = []
        self.short_term_memory: List[MemoryChunk] = []
        self.long_term_memory: List[MemoryChunk] = []

        # 加载历史记忆
        self._load_memory()
        print(f"✅ 优化版记忆引擎初始化完成 | 存储模式：{storage_mode} | 长期记忆：{len(self.long_term_memory)}条")

    def _init_vector_store(self, storage_mode: str, **kwargs) -> BaseVectorStore:
        """初始化向量存储，支持Milvus/Chroma一键切换"""
        collection_name = kwargs.get("collection_name", "brain_memory")
        if storage_mode == "milvus":
            return MilvusVectorStore(
                collection_name=collection_name,
                embeddings=self.embeddings,
                uri=kwargs.get("milvus_uri", os.getenv("MILVUS_URI", "http://localhost:19530")),
                token=kwargs.get("milvus_token", os.getenv("MILVUS_TOKEN", "")),
                dim=kwargs.get("embedding_dim", os.getenv("EMBEDDING_DIM", 384))
            )
        else:
            return ChromaVectorStore(
                collection_name=collection_name,
                embeddings=self.embeddings,
                persist_directory=self.LOCAL_MEMORY_PATH
            )

    def _load_memory(self):
        """从本地加载历史记忆元数据"""
        metadata_path = os.path.join(self.LOCAL_MEMORY_PATH, "memory_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.short_term_memory = [
                MemoryChunk.model_validate_json(json.dumps(m)) for m in metadata.get("short_term_memory", [])
            ]
            self.long_term_memory = [
                MemoryChunk.model_validate_json(json.dumps(m)) for m in metadata.get("long_term_memory", [])
            ]

    def _save_memory(self):
        """保存记忆元数据到本地"""
        os.makedirs(self.LOCAL_MEMORY_PATH, exist_ok=True)
        save_data = {
            "short_term_memory": [json.loads(m.model_dump_json()) for m in self.short_term_memory],
            "long_term_memory": [json.loads(m.model_dump_json()) for m in self.long_term_memory],
            "updated_at": datetime.now().isoformat()
        }
        with open(os.path.join(self.LOCAL_MEMORY_PATH, "memory_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)

    def write_memory(
        self,
        content: str,
        user_query: str = "",
        is_important: bool = False,
        tags: List[str] = None,
        async_insert: bool = False
    ) -> MemoryChunk:
        """
        写入记忆，集成安全脱敏、动态摘要、精细编码全流程
        :param content: 原始内容
        :param user_query: 用户原始提问
        :param is_important: 是否标记为重要记忆
        :param tags: 自定义标签
        :param async_insert: 是否异步插入向量库，优化内存
        :return: 生成的记忆块
        """
        # 1. 安全脱敏，自动过滤敏感信息
        sanitized_content = self.sanitizer.sanitize(content)
        if not sanitized_content.strip():
            raise ValueError("脱敏后内容为空，无法写入记忆")

        # 2. 动态主题预测与多粒度摘要
        topic = self.summarizer.topic_classifier.predict_topic(sanitized_content)
        structured_summary = self.summarizer.generate_summary(sanitized_content, topic=topic)

        # 3. 构建记忆块
        new_memory = MemoryChunk(
            content=sanitized_content,
            structured_summary=structured_summary,
            related_knowledge=self.summarizer.topic_classifier.label_mapping.values(),
            tags=tags if tags else self.summarizer.topic_classifier.predict_topic(sanitized_content).split(","),
            is_important=is_important,
            topic=topic
        )

        # 4. 记忆层级分配
        if is_important:
            new_memory.memory_level = "long_term"
            new_memory.memory_strength = 8.0
            self.long_term_memory.append(new_memory)
            # 写入向量库
            doc = Document(
                page_content=f"{new_memory.structured_summary}\n原始内容：{new_memory.content}",
                metadata=json.loads(new_memory.model_dump_json())
            )
            self.vector_store.add_documents([doc], async_insert=async_insert)
        else:
            self.working_memory.append(new_memory)
            self._trim_working_memory()

        # 5. 持久化保存
        self._save_memory()
        print(f"✅ 记忆写入成功 | ID：{new_memory.chunk_id} | 主题：{topic} | 重要级别：{is_important}")
        return new_memory

    def _trim_working_memory(self):
        """控制工作记忆容量，超限自动转移到短期记忆"""
        total_token = sum([len(chunk.content) for chunk in self.working_memory])
        while total_token > self.WORKING_MEMORY_MAX_TOKEN and len(self.working_memory) > 0:
            moved_memory = self.working_memory.pop(0)
            moved_memory.memory_level = "short_term"
            self.short_term_memory.append(moved_memory)
            total_token = sum([len(chunk.content) for chunk in self.working_memory])
        self._save_memory()

    def retrieve_memory(
        self,
        query: str,
        top_k: int = 5,
        filter_important: bool = False,
        memory_level: str = "all"
    ) -> List[MemoryChunk]:
        """
        增量检索记忆，触发检索效应，自动强化记忆强度
        :param query: 查询文本
        :param top_k: 返回结果数量
        :param filter_important: 仅检索重要记忆
        :param memory_level: 记忆层级过滤
        :return: 匹配的记忆块列表
        """
        # 构建过滤表达式
        filter_expr = "is_important == true" if filter_important else None
        # 向量相似度检索
        docs = self.vector_store.similarity_search(query, top_k=top_k*2, filter_expr=filter_expr)
        semantic_chunk_ids = [doc.metadata.get("chunk_id") for doc in docs]

        # 合并所有记忆候选集
        all_candidate = self.working_memory + self.short_term_memory + self.long_term_memory
        if memory_level != "all":
            all_candidate = [m for m in all_candidate if m.memory_level == memory_level]

        # 多维度打分排序
        matched_memory = []
        for memory in all_candidate:
            tag_match_score = len(set(memory.tags) & set(query.split())) * 0.5
            semantic_score = 0.0
            if memory.chunk_id in semantic_chunk_ids:
                semantic_score = 10 - (semantic_chunk_ids.index(memory.chunk_id) * 0.2)
            total_score = semantic_score + tag_match_score + memory.memory_strength
            memory.match_score = total_score
            matched_memory.append(memory)

        # 排序取top_k
        matched_memory = sorted(matched_memory, key=lambda x: x.match_score, reverse=True)[:top_k]

        # 检索效应：每次提取自动强化记忆强度
        for memory in matched_memory:
            memory.memory_strength = min(10, memory.memory_strength + self.RETRIEVAL_STRENGTH_BOOST)
            memory.last_accessed_at = datetime.now()
            memory.version += 1
            print(f"🔍 记忆提取成功 | ID：{memory.chunk_id} | 强度提升至：{memory.memory_strength:.1f}")

        self._save_memory()
        return matched_memory

    def sleep_consolidation(self):
        """睡眠巩固机制，对应间隔重复、短期转长期记忆"""
        print("\n🌙 启动记忆睡眠巩固机制...")
        now = datetime.now()
        new_long_term_count = 0
        forgotten_count = 0

        # 处理短期记忆
        remaining_short_term = []
        for memory in self.short_term_memory:
            days_since_created = (now - memory.created_at).days
            memory.memory_strength = max(0, memory.memory_strength - days_since_created * self.MEMORY_DECAY_RATE)

            # 符合条件的转移到长期记忆
            if memory.memory_strength >= 5.0 or days_since_created <= self.SHORT_TERM_MEMORY_DAYS / 2:
                memory.memory_level = "long_term"
                self.long_term_memory.append(memory)
                # 写入向量库
                doc = Document(
                    page_content=f"{memory.structured_summary}\n原始内容：{memory.content}",
                    metadata=json.loads(memory.model_dump_json())
                )
                self.vector_store.add_documents([doc])
                new_long_term_count += 1
            # 保留未过期记忆
            elif days_since_created <= self.SHORT_TERM_MEMORY_DAYS and memory.memory_strength > 1.0:
                remaining_short_term.append(memory)
            # 触发遗忘
            else:
                forgotten_count += 1

        self.short_term_memory = remaining_short_term

        # 长期记忆慢衰减
        for memory in self.long_term_memory:
            memory.memory_strength = max(1.0, memory.memory_strength - 0.02 * (now - memory.last_accessed_at).days)

        # 持久化保存
        self._save_memory()
        print(f"🌙 记忆巩固完成 | 新增长期记忆：{new_long_term_count}条 | 清理遗忘：{forgotten_count}条\n")

    def chat(self, user_query: str, is_important: bool = False, chat_history: List[Dict] = None) -> str:
        """带记忆的对话核心入口"""
        # 检索相关记忆
        matched_memories = self.retrieve_memory(user_query, top_k=3)
        # 生成记忆上下文
        context = "\n".join([f"{idx+1}. {m.structured_summary}" for idx, m in enumerate(matched_memories)]) if matched_memories else ""
        # 调用LLM生成回答
        response = self.llm.chat(user_query, context=context, chat_history=chat_history)
        # 写入本次对话到记忆
        self.write_memory(
            content=f"用户提问：{user_query}\n助手回答：{response}",
            user_query=user_query,
            is_important=is_important
        )
        return response
