from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .base_llm import BaseLLM

class OpenAILLM(BaseLLM):
    """OpenAI兼容接口实现，向下兼容"""
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.3,
        **kwargs
    ):
        self.model = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )

    def chat(self, user_input: str, context: str = "", chat_history: List[Dict] = None) -> str:
        memory_context = f"【相关历史记忆】：\n{context}" if context else "无相关历史记忆"
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是专业的知识助手，结合用户提供的历史记忆回答问题，逻辑清晰，简洁准确。\n{memory_context}"),
            ("human", "{user_input}")
        ])
        chain = prompt | self.model
        return chain.invoke({"memory_context": memory_context, "user_input": user_input}).content

    def encode(self, text: str) -> str:
        prompt = f"对以下内容进行核心信息提纯，保留关键知识点、逻辑和适用场景：\n{text}"
        return self.model.invoke(prompt).content
