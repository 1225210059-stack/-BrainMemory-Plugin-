import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from .topic_classifier import TopicClassifier

class DynamicSummarizer:
    """基于主题的多粒度动态摘要生成器，替换原有简单编码逻辑"""
    def __init__(self, device: str = "auto"):
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        # 摘要生成模型，支持中英文，效果优于原方案的bart-large-cnn
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)
        self.model.eval()
        # 主题分类器
        self.topic_classifier = TopicClassifier(device=device)

    def generate_summary(
        self,
        text: str,
        topic: str = None,
        custom_max_length: int = None,
        min_length: int = 30
    ) -> str:
        """
        生成动态摘要
        :param text: 待摘要的原始文本
        :param topic: 手动指定主题，不指定则自动预测
        :param custom_max_length: 自定义摘要最大长度
        :param min_length: 摘要最小长度
        :return: 生成的摘要文本
        """
        # 自动预测主题
        if not topic:
            topic = self.topic_classifier.predict_topic(text)
        
        # 确定摘要长度
        max_length = custom_max_length if custom_max_length else self.topic_classifier.get_summary_max_length(topic)
        
        # 文本编码
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成摘要
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids,
                num_beams=4,
                max_length=max_length,
                min_length=min_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2
            )
        
        # 解码返回
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
