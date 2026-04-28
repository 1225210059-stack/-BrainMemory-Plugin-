import os
import json
from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from .storage import BaseStorage, LocalStorage, CloudStorage, HybridStorage
from .encryption import Encryptor

load_dotenv()

class MemoryChunk(BaseModel):
    """记忆块核心数据模型，对应人脑记忆单元"""
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

class BrainMemory:
    """类脑长记忆核心引擎"""
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        storage_mode: str = "local",
        **kwargs
    ):
        """
        初始化记忆引擎
        :param llm: 大模型实例，支持所有LangChain兼容的大模型
        :param embeddings: 嵌入模型实例
        :param storage_mode: 存储模式，local/cloud/hybrid
        :param kwargs: 其他配置参数，对应.env中的配置
        """
        self.llm = llm
        self.embeddings = embeddings
        self.storage_mode = storage_mode

        # 加载核心参数
        self.WORKING_MEMORY_MAX_TOKEN = int(kwargs.get("working_memory_max_token", os.getenv("WORKING_MEMORY_MAX_TOKEN", 4000)))
        self.SHORT_TERM_MEMORY_DAYS = int(kwargs.get("short_term_memory_days", os.getenv("SHORT_TERM_MEMORY_DAYS", 14)))
        self.MEMORY_DECAY_RATE = float(kwargs.get("memory_decay_rate", os.getenv("MEMORY_DECAY_RATE", 0.1)))
        self.RETRIEVAL_STRENGTH_BOOST = float(kwargs.get("retrieval_strength_boost", os.getenv("RETRIEVAL_STRENGTH_BOOST", 0.8)))

        # 初始化加密器
        self.encryptor = Encryptor(
            password=kwargs.get("encryption_password", os.getenv("ENCRYPTION_PASSWORD", "default_password").encode()),
            salt=kwargs.get("encryption_salt", os.getenv("ENCRYPTION_SALT", "default_salt").encode())
        )

        # 初始化存储处理器
        self.storage = self._init_storage(**kwargs)
        print(f"✅ 记忆引擎初始化完成，存储模式：{self.storage_mode}")

        # 三级记忆分层
        self.working_memory: List[MemoryChunk] = []
        self.short_term_memory: List[MemoryChunk] = []
        self.long_term_memory: List[MemoryChunk] = []

        # 加载历史记忆
        self._load_memory()
        self.vector_db = self.storage.get_vector_db()
        print(f"✅ 历史记忆加载完成：长期记忆{len(self.long_term_memory)}条，短期记忆{len(self.short_term_memory)}条")

    def _init_storage(self, **kwargs) -> BaseStorage:
        """根据存储模式初始化存储处理器"""
        if self.storage_mode == "local":
            return LocalStorage(
                embeddings=self.embeddings,
                local_path=kwargs.get("local_path", os.getenv("LOCAL_MEMORY_PATH", "./local_memory_db"))
            )
        elif self.storage_mode == "cloud":
            return CloudStorage(
                embeddings=self.embeddings,
                access_key=kwargs.get("cloud_access_key", os.getenv("CLOUD_ACCESS_KEY", "")),
                secret_key=kwargs.get("cloud_secret_key", os.getenv("CLOUD_SECRET_KEY", "")),
                endpoint_url=kwargs.get("cloud_endpoint_url", os.getenv("CLOUD_ENDPOINT_URL", "")),
                region=kwargs.get("cloud_region", os.getenv("CLOUD_REGION", "auto")),
                bucket_name=kwargs.get("cloud_bucket_name", os.getenv("CLOUD_BUCKET_NAME", "")),
                encryptor=self.encryptor
            )
        elif self.storage_mode == "hybrid":
            local_storage = LocalStorage(
                embeddings=self.embeddings,
                local_path=kwargs.get("local_path", os.getenv("LOCAL_MEMORY_PATH", "./local_memory_db"))
            )
            cloud_storage = CloudStorage(
                embeddings=self.embeddings,
                access_key=kwargs.get("cloud_access_key", os.getenv("CLOUD_ACCESS_KEY", "")),
                secret_key=kwargs.get("cloud_secret_key", os.getenv("CLOUD_SECRET_KEY", "")),
                endpoint_url=kwargs.get("cloud_endpoint_url", os.getenv("CLOUD_ENDPOINT_URL", "")),
                region=kwargs.get("cloud_region", os.getenv("CLOUD_REGION", "auto")),
                bucket_name=kwargs.get("cloud_bucket_name", os.getenv("CLOUD_BUCKET_NAME", "")),
                encryptor=self.encryptor
            )
            return HybridStorage(local_storage, cloud_storage)
        else:
            raise ValueError(f"不支持的存储模式：{self.storage_mode}，可选值：local/cloud/hybrid")

    def _load_memory(self):
        """从存储中加载历史记忆"""
        metadata = self.storage.load_metadata()
        # 反序列化记忆块
        self.short_term_memory = [
            MemoryChunk.model_validate_json(json.dumps(m)) for m in metadata.get("short_term_memory", [])
        ]
        self.long_term_memory = [
            MemoryChunk.model_validate_json(json.dumps(m)) for m in metadata.get("long_term_memory", [])
        ]

    def _save_memory(self):
        """保存记忆到底层存储"""
        save_data = {
            "short_term_memory": [json.loads(m.model_dump_json()) for m in self.short_term_memory],
            "long_term_memory": [json.loads(m.model_dump_json()) for m in self.long_term_memory],
            "updated_at": datetime.now().isoformat()
        }
        self.storage.save_metadata(save_data)

    def _fine_encoding(self, content: str, user_query: str = "") -> MemoryChunk:
        """精细编码引擎，对应费曼学习法、生成效应、结构化编码"""
        encoding_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是专业的记忆编码专家，严格遵循以下规则对内容进行精细编码：
            1. 核心提纯：过滤冗余、无效、幻觉内容，提取核心知识点、结论、关键信息
            2. 结构化压缩：用【核心定义】【实操方法】【适用场景】【易错点】4个维度结构化总结，不超过300字
            3. 费曼转化：把专业内容转化为无术语的通俗核心逻辑，确保零基础可理解
            4. 关联挖掘：提炼3个核心关键词，作为关联知识点标签
            输出格式严格按照以下格式，不要添加额外内容：
            【结构化摘要】：xxx
            【通俗核心】：xxx
            【核心关键词】：关键词1,关键词2,关键词3
            """),
            ("human", "待编码内容：{content}\n用户原始提问：{user_query}")
        ])
        encoding_chain = encoding_prompt | self.llm
        encoding_result = encoding_chain.invoke({"content": content, "user_query": user_query}).content

        # 解析编码结果，容错处理
        try:
            summary_part = encoding_result.split("【结构化摘要】：")[1].split("【通俗核心】：")[0].strip()
            core_part = encoding_result.split("【通俗核心】：")[1].split("【核心关键词】：")[0].strip()
            keywords = [k.strip() for k in encoding_result.split("【核心关键词】：")[1].strip().split(",")]
        except Exception:
            summary_part = content[:300]
            core_part = content
            keywords = []

        return MemoryChunk(
            content=content,
            structured_summary=f"{summary_part}\n通俗核心：{core_part}",
            related_knowledge=keywords,
            tags=keywords
        )

    def write_memory(self, content: str, user_query: str = "", is_important: bool = False) -> MemoryChunk:
        """写入记忆，核心入口"""
        # 精细编码
        new_memory = self._fine_encoding(content, user_query)
        new_memory.is_important = is_important

        # 重要内容直接进入长期记忆
        if is_important:
            new_memory.memory_level = "long_term"
            new_memory.memory_strength = 8.0
            self.long_term_memory.append(new_memory)
            # 写入向量存储
            doc = Document(
                page_content=f"{new_memory.structured_summary}\n原始内容：{new_memory.content}",
                metadata={
                    "chunk_id": new_memory.chunk_id,
                    "tags": ",".join(new_memory.tags),
                    "memory_strength": new_memory.memory_strength,
                    "is_important": is_important,
                    "created_at": new_memory.created_at.isoformat()
                }
            )
            self.storage.save_vector_doc(doc)
        else:
            # 普通内容进入工作记忆
            self.working_memory.append(new_memory)
            self._trim_working_memory()

        # 持久化保存
        self._save_memory()
        print(f"✅ 记忆写入成功 | ID：{new_memory.chunk_id} | 重要级别：{is_important} | 关键词：{new_memory.tags}")
        return new_memory

    def _trim_working_memory(self):
        """控制工作记忆容量，对应人脑工作记忆4±2组块限制"""
        total_token = sum([len(chunk.content) for chunk in self.working_memory])
        while total_token > self.WORKING_MEMORY_MAX_TOKEN and len(self.working_memory) > 0:
            moved_memory = self.working_memory.pop(0)
            moved_memory.memory_level = "short_term"
            self.short_term_memory.append(moved_memory)
            total_token = sum([len(chunk.content) for chunk in self.working_memory])
        self._save_memory()

    def retrieve_memory(self, query: str, top_k: int = 5) -> List[MemoryChunk]:
        """检索记忆，核心入口，对应主动提取练习、检索效应"""
        # 多线索混合检索
        semantic_docs = self.vector_db.similarity_search(query, k=top_k*2)
        semantic_chunk_ids = [doc.metadata["chunk_id"] for doc in semantic_docs]

        # 合并所有记忆候选集
        all_candidate = self.working_memory + self.short_term_memory + self.long_term_memory
        matched_memory = []

        for memory in all_candidate:
            # 多维度打分
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
            print(f"🔍 记忆提取成功 | ID：{memory.chunk_id} | 记忆强度提升至：{memory.memory_strength:.1f}")

        self._save_memory()
        return matched_memory

    def sleep_consolidation(self):
        """睡眠巩固机制，对应人脑睡眠记忆巩固、间隔重复系统"""
        print("\n🌙 启动记忆睡眠巩固机制...")
        now = datetime.now()
        new_long_term_count = 0
        forgotten_count = 0

        # 处理短期记忆
        remaining_short_term = []
        for memory in self.short_term_memory:
            days_since_created = (now - memory.created_at).days
            # 记忆强度自然衰减
            memory.memory_strength = max(0, memory.memory_strength - days_since_created * self.MEMORY_DECAY_RATE)

            # 符合条件的转移到长期记忆
            if memory.memory_strength >= 5.0 or days_since_created <= self.SHORT_TERM_MEMORY_DAYS / 2:
                memory.memory_level = "long_term"
                self.long_term_memory.append(memory)
                # 写入向量存储
                doc = Document(
                    page_content=f"{memory.structured_summary}\n原始内容：{memory.content}",
                    metadata={
                        "chunk_id": memory.chunk_id,
                        "tags": ",".join(memory.tags),
                        "memory_strength": memory.memory_strength,
                        "is_important": memory.is_important,
                        "created_at": memory.created_at.isoformat()
                    }
                )
                self.storage.save_vector_doc(doc)
                new_long_term_count += 1
            # 保留未过期的中强度记忆
            elif days_since_created <= self.SHORT_TERM_MEMORY_DAYS and memory.memory_strength > 1.0:
                remaining_short_term.append(memory)
            # 触发遗忘
            else:
                forgotten_count += 1

        # 更新短期记忆池
        self.short_term_memory = remaining_short_term

        # 长期记忆极慢衰减
        for memory in self.long_term_memory:
            memory.memory_strength = max(1.0, memory.memory_strength - 0.02 * (now - memory.last_accessed_at).days)

        # 持久化保存
        self._save_memory()
        print(f"🌙 记忆巩固完成 | 新增长期记忆：{new_long_term_count}条 | 清理遗忘记忆：{forgotten_count}条\n")

    def chat(self, user_query: str, is_important: bool = False) -> str:
        """带记忆的对话入口，一键使用"""
        # 检索相关记忆
        matched_memories = self.retrieve_memory(user_query)
        # 构建记忆上下文
        memory_context = "【相关历史记忆】：\n"
        if matched_memories:
            for idx, memory in enumerate(matched_memories, 1):
                memory_context += f"{idx}. {memory.structured_summary}\n"
        else:
            memory_context += "无相关历史记忆\n"

        # 生成回答
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是专业的知识助手，严格结合用户提供的历史记忆回答问题，逻辑清晰，简洁准确，不重复已明确的知识点。\n{memory_context}"),
            ("human", "{user_query}")
        ])
        chat_chain = chat_prompt | self.llm
        response = chat_chain.invoke({
            "memory_context": memory_context,
            "user_query": user_query
        }).content

        # 写入本次对话到记忆
        self.write_memory(
            content=f"用户提问：{user_query}\n助手回答：{response}",
            user_query=user_query,
            is_important=is_important
        )

        return response
