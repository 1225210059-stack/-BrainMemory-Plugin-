import pytest
from src import OptimizedBrainMemory, Llama3LLM, MilvusVectorStore

@pytest.mark.integration
def test_hybrid_model_full_flow():
    """测试LLaMA-3混合架构全链路"""
    # 初始化LLaMA-3模型
    llm = Llama3LLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_8bit=True
    )
    # 初始化记忆引擎
    memory = OptimizedBrainMemory(
        llm=llm,
        storage_mode="milvus"
    )
    # 写入记忆
    memory.write_memory(
        "机器学习是人工智能的核心，是使计算机具有智能的根本途径，其应用遍及人工智能的各个领域，它主要使用归纳、综合而不是演绎。",
        is_important=True,
        tags=["人工智能", "机器学习"]
    )
    # 检索记忆
    memories = memory.retrieve_memory("什么是机器学习？", top_k=1)
    assert len(memories) == 1
    # 带记忆对话
    response = memory.chat("什么是机器学习？")
    assert "人工智能" in response
    # 记忆巩固
    memory.sleep_consolidation()
    print("✅ 混合架构全链路测试通过")
