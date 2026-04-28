"""
BrainMemory-Plugin 类脑长记忆插件
基于人脑记忆原理打造的大模型长记忆系统，支持本地/云端/混合存储一键切换
"""
from .core import BrainMemory
from .storage import LocalStorage, CloudStorage, HybridStorage
from .encryption import Encryptor

__version__ = "1.0.0"
__all__ = ["BrainMemory", "LocalStorage", "CloudStorage", "HybridStorage", "Encryptor"]
