import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class Encryptor:
    """端到端加密器，用于云端存储数据加密，密钥仅使用者持有"""
    def __init__(self, password: bytes, salt: bytes):
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000
        )
        self.key = base64.urlsafe_b64encode(self.kdf.derive(password))
        self.cipher = Fernet(self.key)

    def encrypt(self, data: dict) -> bytes:
        """加密字典数据，返回密文"""
        import json
        json_str = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        return self.cipher.encrypt(json_str)

    def decrypt(self, encrypted_data: bytes) -> dict:
        """解密密文，返回原始字典数据"""
        import json
        json_str = self.cipher.decrypt(encrypted_data).decode("utf-8")
        return json.loads(json_str)
