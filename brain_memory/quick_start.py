"""
BrainMemory-Plugin 快速上手示例
运行前请先复制.env.example为.env，配置好你的大模型参数
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from brain_memory import BrainMemory

# 加载环境变量
load_dotenv()

if __name__ == "__main__":
    # 1. 初始化大模型和嵌入模型（可替换为本地开源Ollama模型）
    llm = ChatOpenAI(
        model_name=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.3
    )
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    # 2. 初始化类脑记忆引擎，自动读取.env中的存储模式配置
    memory_engine = BrainMemory(
        llm=llm,
        embeddings=embeddings,
        storage_mode=os.getenv("STORAGE_MODE", "local")
    )

    # 3. 带记忆的对话测试
    print("===== 类脑长记忆插件 快速测试 =====")
    # 第一轮对话，写入记忆
    print("\n👤 你：什么是主动提取练习？它和被动重读有什么区别？")
    res1 = memory_engine.chat("什么是主动提取练习？它和被动重读有什么区别？", is_important=True)
    print(f"🤖 助手：{res1}")

    # 第二轮对话，自动调用历史记忆
    print("\n👤 你：这个方法怎么用在考研英语背单词上？")
    res2 = memory_engine.chat("这个方法怎么用在考研英语背单词上？")
    print(f"🤖 助手：{res2}")

    # 执行记忆巩固
    memory_engine.sleep_consolidation()

    # 第三轮对话，验证长期记忆
    print("\n👤 你：怎么避免学习中的熟练度错觉？")
    res3 = memory_engine.chat("怎么避免学习中的熟练度错觉？")
    print(f"🤖 助手：{res3}")

    # 开启无限对话
    while True:
        user_input = input("\n👤 请输入你的问题（输入exit退出）：")
        if user_input.lower() == "exit":
            print("👋 对话结束，所有记忆已按配置保存")
            break
        is_important = input("是否标记为重要记忆？(y/n)：").lower() == "y"
        res = memory_engine.chat(user_input, is_important)
        print(f"🤖 助手：{res}")
