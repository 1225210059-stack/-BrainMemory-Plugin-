"""
基础功能测试用例
运行命令：python -m pytest tests/test_basic.py -v
"""
import pytest
import tempfile
from brain_memory import MemoryChunk, Encryptor, LocalStorage
from langchain_openai import OpenAIEmbeddings

def test_encryptor():
    """测试加密解密功能"""
    encryptor = Encryptor(b"test_password", b"test_salt")
    test_data = {"key": "value", "number": 123}
    encrypted = encryptor.encrypt(test_data)
    decrypted = encryptor.decrypt(encrypted)
    assert decrypted["key"] == "value"
    assert decrypted["number"] == 123

def test_memory_chunk():
    """测试记忆块模型"""
    chunk = MemoryChunk(content="测试内容", structured_summary="测试摘要")
    assert chunk.chunk_id is not None
    assert chunk.memory_strength == 1.0
    assert chunk.memory_level == "short_term"

def test_local_storage():
    """测试本地存储功能"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = OpenAIEmbeddings(api_key="test_key", base_url="https://test.com")
        storage = LocalStorage(embeddings, local_path=tmpdir)
        test_meta = {"short_term_memory": [], "long_term_memory": [], "updated_at": "2024-01-01"}
        storage.save_metadata(test_meta)
        loaded_meta = storage.load_metadata()
        assert loaded_meta["updated_at"] == test_meta["updated_at"]
        assert storage.is_connected() == True
