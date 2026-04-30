import pytest
from src import SecuritySanitizer

def test_pii_sanitization():
    """测试PII信息脱敏"""
    sanitizer = SecuritySanitizer()
    test_text = "我的手机号是13812345678，身份证号是110101199001011234，银行卡号是6222021234567890123，邮箱是test@example.com"
    sanitized = sanitizer.sanitize(test_text)
    assert "13812345678" not in sanitized
    assert "110101199001011234" not in sanitized
    assert "6222021234567890123" not in sanitized
    assert "test@example.com" not in sanitized
    assert "****" in sanitized
    print("✅ PII脱敏测试通过")

def test_sensitive_keyword_sanitization():
    """测试敏感关键词脱敏"""
    sanitizer = SecuritySanitizer()
    test_text = "我的API密钥是sk-1234567890，密码是123456"
    sanitized = sanitizer.sanitize(test_text)
    assert "API密钥" not in sanitized
    assert "密码" not in sanitized
    print("✅ 敏感关键词脱敏测试通过")
