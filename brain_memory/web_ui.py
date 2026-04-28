"""
BrainMemory-Plugin 可视化Web界面
启动命令：streamlit run examples/web_ui.py
"""
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from brain_memory import BrainMemory

# 页面配置
st.set_page_config(
    page_title="类脑长记忆插件",
    page_icon="🧠",
    layout="wide"
)

# 加载环境变量
load_dotenv()

# 初始化会话状态
if "memory_engine" not in st.session_state:
    st.session_state.memory_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏配置
with st.sidebar:
    st.title("🧠 类脑长记忆插件")
    st.divider()

    # 大模型配置
    st.subheader("大模型配置")
    llm_type = st.radio("模型类型", ["OpenAI兼容", "本地Ollama"], index=0)
    if llm_type == "OpenAI兼容":
        llm_model = st.text_input("模型名称", value=os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
        api_key = st.text_input("API密钥", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        base_url = st.text_input("API地址", value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        embedding_model = st.text_input("嵌入模型", value=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    else:
        llm_model = st.text_input("Ollama模型名称", value=os.getenv("LLM_MODEL", "qwen2:7b"))
        ollama_base_url = st.text_input("Ollama地址", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        embedding_model = st.text_input("嵌入模型", value=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))

    # 存储模式配置
    st.subheader("存储模式")
    storage_mode = st.selectbox("选择存储模式", ["local", "cloud", "hybrid"], index=0)
    is_important = st.checkbox("默认标记对话为重要记忆", value=False)

    # 初始化按钮
    if st.button("初始化记忆引擎", type="primary", use_container_width=True):
        with st.spinner("正在初始化记忆引擎..."):
            # 初始化大模型
            if llm_type == "OpenAI兼容":
                llm = ChatOpenAI(
                    model_name=llm_model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.3
                )
                embeddings = OpenAIEmbeddings(
                    model=embedding_model,
                    api_key=api_key,
                    base_url=base_url
                )
            else:
                llm = ChatOllama(
                    model=llm_model,
                    base_url=ollama_base_url,
                    temperature=0.3
                )
                embeddings = OllamaEmbeddings(
                    model=embedding_model,
                    base_url=ollama_base_url
                )
            # 初始化记忆引擎
            st.session_state.memory_engine = BrainMemory(
                llm=llm,
                embeddings=embeddings,
                storage_mode=storage_mode
            )
            st.success("✅ 记忆引擎初始化完成！")

    # 记忆巩固按钮
    if st.session_state.memory_engine and st.button("执行记忆睡眠巩固", use_container_width=True):
        with st.spinner("正在执行记忆巩固..."):
            st.session_state.memory_engine.sleep_consolidation()
            st.success("🌙 记忆巩固完成！")

    # 记忆统计
    if st.session_state.memory_engine:
        st.divider()
        st.subheader("记忆统计")
        st.write(f"工作记忆：{len(st.session_state.memory_engine.working_memory)} 条")
        st.write(f"短期记忆：{len(st.session_state.memory_engine.short_term_memory)} 条")
        st.write(f"长期记忆：{len(st.session_state.memory_engine.long_term_memory)} 条")

# 主界面对话区
st.header("类脑长记忆对话系统")
st.caption("基于人脑记忆原理打造的大模型长记忆系统，支持本地/云端/混合存储一键切换")

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 对话输入
if prompt := st.chat_input("请输入你的问题..."):
    if not st.session_state.memory_engine:
        st.error("请先在侧边栏初始化记忆引擎！")
    else:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在思考并检索记忆..."):
                response = st.session_state.memory_engine.chat(prompt, is_important=is_important)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# 清空对话按钮
if st.button("清空当前对话", use_container_width=True):
    st.session_state.messages = []
    st.rerun()
