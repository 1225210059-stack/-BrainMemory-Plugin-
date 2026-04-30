from src import OptimizedBrainMemory, Llama3LLM

if __name__ == "__main__":
    print("===== LLaMA-3 混合架构记忆系统 =====")
    # 初始化本地LLaMA-3模型
    llama3 = Llama3LLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_8bit=True,
        max_new_tokens=1024
    )
    # 初始化记忆引擎
    memory_engine = OptimizedBrainMemory(llm=llama3, storage_mode="local")

    # 无限对话
    while True:
        user_input = input("\n👤 请输入你的问题（输入exit退出）：")
        if user_input.lower() == "exit":
            print("👋 对话结束")
            break
        is_important = input("是否标记为重要记忆？(y/n)：").lower() == "y"
        res = memory_engine.chat(user_input, is_important=is_important)
        print(f"🤖 助手：{res}")
