from Cryptodome.Cipher import AES
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Protocol.KDF import HKDF
from Cryptodome.Hash import HMAC, SHA256
from Cryptodome.Random import get_random_bytes
from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP
import base64
import uuid
import secrets


def generate_collision_resistant_uuid() -> str:
    """
    Generate collision-resistant UUIDv4 format random string
    - Uses cryptographically secure random number generator (secrets)
    - Complies with UUIDv4 standard (version identifier and variant bits)
    - Extremely low collision probability (approximately 1/2^122)
    """
    # Generate 16 bytes (128 bits) of random data
    random_bytes = secrets.token_bytes(16)
    
    # Set UUIDv4 version and variant bits
    # Version bits: Set bits 6 to 8 to 0b0100 (corresponding to UUIDv4)
    # Variant bits: Set bits 64 to 66 to 0b10 (RFC 4122 standard)
    random_bytes = bytearray(random_bytes)
    random_bytes[6] = (random_bytes[6] & 0x0F) | 0x40  # Set version bits
    random_bytes[8] = (random_bytes[8] & 0x3F) | 0x80  # Set variant bits
    
    # Convert to UUID object and format as string
    return str(uuid.UUID(bytes=bytes(random_bytes)))

class AESHelper:
    @staticmethod
    def generate_key(key_length=16):
        """Generate AES standard compliant key (16/24/32 bytes)"""
        return get_random_bytes(key_length)
    
    @staticmethod
    def generate_iv():
        """Generate random initialization vector required for CBC mode"""
        return get_random_bytes(AES.block_size)
    
    @staticmethod
    def pad_data(data):
        """PKCS7 padding"""
        return pad(data, AES.block_size)
    
    @staticmethod
    def unpad_data(padded_data):
        """PKCS7 unpadding"""
        return unpad(padded_data, AES.block_size)

class AESProcessor:
    """Encryption/decryption processor (supports ECB/CBC modes)"""
    def __init__(self, key, mode=AES.MODE_CBC, iv=None):
        self.key = key
        self.mode = mode
        self.iv = iv  # Allow manual IV input
    
    def _build_cipher(self):
        """Create cipher based on mode"""
        if self.mode == AES.MODE_CBC:
            if self.iv is None:
                raise ValueError("CBC mode requires IV")
            return AES.new(self.key, self.mode, iv=self.iv)
        return AES.new(self.key, self.mode)  # ECB mode

class Encryptor(AESProcessor):
    """Encryption processor"""
    def encrypt(self, plaintext):
        padded_data = AESHelper.pad_data(plaintext)
        cipher = self._build_cipher()
        ciphertext = cipher.encrypt(padded_data)
        
        # Return structure: (IV or None, base64 ciphertext)
        return (
            self.iv if self.mode == AES.MODE_CBC else None,
            base64.b64encode(ciphertext).decode('utf-8')
        )

class Decryptor(AESProcessor):
    """Decryption processor"""
    def decrypt(self, ciphertext_base64):
        ciphertext = base64.b64decode(ciphertext_base64)
        cipher = self._build_cipher()
        decrypted_data = cipher.decrypt(ciphertext)
        return AESHelper.unpad_data(decrypted_data)

if __name__ == "__main__":
    # Generate key
    key = AESHelper.generate_key(16)
    plaintext = b"Mr.Brown, a male patient, reported noticing a lump under his left nipple a few weeks ago that is painful to touch and about the size of a quarter, along with stomach pains that cause immediate fullness and extreme pain, preventing him from eating. The doctor advised that the lump under the nipple be removed and biopsied for diagnosis and treatment, and also recommended that Mr.Brown undergo upper GI endoscopy as soon as possible as the stomach problem could be an ulcer with an issue at the pylorus, emphasizing the importance of an exact diagnosis to start appropriate therapy without delay."  # Your original text
    text = b"Mr.Brown, a male patient, reported noticing a lump under his left nipple a few weeks ago that is painful to touch and about the size of a quarter, along with stomach pains that cause immediate fullness and extreme pain, preventing him from eating. The doctor advised that the lump under the nipple be removed and biopsied for diagnosis and treatment, and also recommended that Mr.Brown undergo upper GI endoscopy as soon as possible as the stomach problem could be an ulcer with an issue at the pylorus, emphasizing the importance of an exact diagnosis to start appropriate therapy without delay."
    # -------------------- CBC mode test --------------------
    # Encryption
    iv = AESHelper.generate_iv()  # Generate independent IV
    print(iv)

    iv_t = base64.b64encode(iv).decode()
    print(isinstance(iv_t, str))
    # print(base64.b64decode(base64.b64encode(iv)))
    cbc_encryptor = Encryptor(key, mode=AES.MODE_CBC, iv=iv)
    _, cbc_ciphertext = cbc_encryptor.encrypt(iv)
    print(f"CBC encryption result: {cbc_ciphertext}")
    # print(len(iv))
    # print(base64.b64encode(iv).decode('utf-8'))
    # Decryption (requires IV)
    cbc_decryptor = Decryptor(key, mode=AES.MODE_CBC, iv=iv)
    cbc_plaintext = cbc_decryptor.decrypt(cbc_ciphertext)
    print(f"CBC decryption result: {cbc_plaintext}")

    # -------------------- ECB mode test --------------------
    # Encryption
    # ecb_encryptor = Encryptor(key, mode=AES.MODE_ECB)
    # _, ecb_ciphertext = ecb_encryptor.encrypt(plaintext)
    # print(f"ECB encryption result: {ecb_ciphertext}")
    # # print(len(AESHelper.pad_data(text)))
    # # print(len(AESHelper.pad_data(plaintext)))
    # # print(len(base64.b64decode(cbc_ciphertext)))
    # # print(len(base64.b64decode(ecb_ciphertext)))
    # # Decryption
    # ecb_decryptor = Decryptor(key, mode=AES.MODE_ECB)
    # ecb_plaintext = ecb_decryptor.decrypt(ecb_ciphertext)
    # print(f"ECB decryption result: {ecb_plaintext.decode('utf-8')}")
