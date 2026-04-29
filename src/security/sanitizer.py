import re
from typing import List, Set

class SecuritySanitizer:
    """安全脱敏过滤器，自动过滤敏感信息，集成到记忆写入全流程"""
    # 默认敏感关键词
    DEFAULT_SENSITIVE_KEYWORDS: Set[str] = {
        "password", "passwd", "pwd", "api_key", "apikey", "secret_key",
        "secret", "token", "credit card", "creditcard", "银行卡", "密码",
        "私钥", "密钥", "身份证", "ssn", "手机号", "电话", "邮箱", "email"
    }
    # PII正则规则，覆盖国内外常见个人信息
    PII_PATTERNS: List[re.Pattern] = [
        # 国内身份证号
        re.compile(r"\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]\b"),
        # 国内手机号
        re.compile(r"\b1[3-9]\d{9}\b"),
        # 银行卡号
        re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4,10}\b"),
        # 邮箱
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        # 美国SSN
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        # 信用卡号
        re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b")
    ]
    # 替换掩码
    MASK = "****"

    def __init__(self, custom_sensitive_keywords: List[str] = None):
        self.sensitive_keywords = self.DEFAULT_SENSITIVE_KEYWORDS.copy()
        if custom_sensitive_keywords:
            self.sensitive_keywords.update([kw.lower() for kw in custom_sensitive_keywords])

    def sanitize(self, text: str) -> str:
        """
        文本脱敏主方法
        :param text: 待脱敏的原始文本
        :return: 脱敏后的文本
        """
        if not text:
            return text
        
        # 1. 替换PII个人信息
        for pattern in self.PII_PATTERNS:
            text = pattern.sub(self.MASK, text)
        
        # 2. 替换敏感关键词
        text_lower = text.lower()
        for kw in self.sensitive_keywords:
            if kw in text_lower:
                # 不区分大小写替换，保留原文格式
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                text = pattern.sub(self.MASK * len(kw), text)
        
        return text
