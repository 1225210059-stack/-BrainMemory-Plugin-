import torch
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from .base_llm import BaseLLM

class Llama3LLM(BaseLLM):
    """LLaMA-3本地大模型，支持8bit/4bit量化，混合架构核心"""
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # 量化配置，降低显存占用
        bnb_config = None
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        # 加载模型与分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()

    def _build_prompt(self, user_input: str, context: str = "", chat_history: List[Dict] = None) -> str:
        """构建LLaMA-3对话格式prompt，集成记忆上下文"""
        messages = []
        # 系统提示词，集成记忆上下文
        system_prompt = "你是专业的知识助手，结合用户提供的相关历史信息回答问题，逻辑清晰，简洁准确，不重复已明确的知识点。"
        if context:
            system_prompt += f"\n\n以下是相关历史信息：\n{context}"
        messages.append({"role": "system", "content": system_prompt})

        # 历史对话
        if chat_history:
            messages.extend(chat_history)
        
        # 当前用户输入
        messages.append({"role": "user", "content": user_input})

        # 转换为LLaMA-3格式
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def chat(self, user_input: str, context: str = "", chat_history: List[Dict] = None) -> str:
        """带记忆的对话核心接口"""
        prompt = self._build_prompt(user_input, context, chat_history)
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192  # 支持8k上下文，可扩展
        ).to(self.model.device)

        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 解码返回，过滤输入prompt
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()

    def encode(self, text: str) -> str:
        """文本精细编码，用于记忆生成"""
        prompt = f"对以下内容进行核心信息提纯，保留关键知识点、逻辑和适用场景：\n{text}"
        return self.chat(prompt)
