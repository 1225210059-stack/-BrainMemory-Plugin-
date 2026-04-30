import time
import pytest
from src import OptimizedBrainMemory, OpenAILLM, OpenAIEmbeddings

@pytest.mark.performance
def test_milvus_search_speed():
    """测试Milvus检索速度，要求<500ms"""
    llm = OpenAILLM(api_key="test", base_url="https://test.com")
    embeddings = OpenAIEmbeddings(api_key="test", base_url="https://test.com")
    memory = OptimizedBrainMemory(
        llm=llm,
        embeddings=embeddings,
        storage_mode="milvus",
        milvus_uri="http://localhost:19530"
    )
    # 预热
    memory.retrieve_memory("测试", top_k=5)
    # 性能测试
    start_time = time.time()
    memory.retrieve_memory("人工智能的核心技术", top_k=10)
    elapsed = time.time() - start_time
    assert elapsed < 0.5, f"检索速度过慢：{elapsed*1000}ms"
    print(f"✅ Milvus检索速度：{elapsed*1000:.2f}ms")

@pytest.mark.performance
def test_summary_generation_speed():
    """测试动态摘要生成速度"""
    from src import DynamicSummarizer
    summarizer = DynamicSummarizer()
    test_text = "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。深度学习是机器学习中一种基于对数据进行表征学习的方法。观测值（例如一幅图像）可以使用多种方式来表示，如每个像素强度值的向量，或者更抽象地表示成一系列边、特定形状的区域等。而使用某些特定的表示方法更容易从实例中学习任务（例如，人脸识别或面部表情识别）。深度学习的好处是用非监督式或半监督式的特征学习和分层特征提取高效算法，来替代手工获取特征。"
    start_time = time.time()
    summary = summarizer.generate_summary(test_text, topic="technology")
    elapsed = time.time() - start_time
    assert len(summary) > 50
    assert elapsed < 2.0
    print(f"✅ 摘要生成速度：{elapsed*1000:.2f}ms，摘要长度：{len(summary)}字")
