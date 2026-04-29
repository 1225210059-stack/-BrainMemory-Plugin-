import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TopicClassifier:
    """主题分类器，使用专业主题分类模型，修复原方案SNLI模型错误"""
    def __init__(self, device: str = "auto"):
        # 专业多主题分类模型，支持21个通用主题
        self.model_name = "cardiffnlp/tweet-topic-21-multi"
        self.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # 主题标签映射，覆盖学习、工作、生活全场景
        self.label_mapping = {
            0: "daily", 1: "sports", 2: "technology", 3: "finance",
            4: "politics", 5: "entertainment", 6: "health", 7: "education",
            8: "travel", 9: "food", 10: "fashion", 11: "gaming",
            12: "music", 13: "culture", 14: "science", 15: "business",
            16: "law", 17: "environment", 18: "lifestyle", 19: "other"
        }
        # 主题-摘要长度映射，技术/教育类保留更多细节
        self.topic_max_length = {
            "technology": 200, "science": 180, "education": 150,
            "finance": 150, "business": 150, "law": 180,
            "health": 120, "daily": 100, "entertainment": 80,
            "default": 100
        }

    def predict_topic(self, text: str) -> str:
        """预测文本主题"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            return self.label_mapping.get(predicted_label, "default")

    def get_summary_max_length(self, topic: str) -> int:
        """根据主题获取摘要最优长度"""
        return self.topic_max_length.get(topic, self.topic_max_length["default"])
