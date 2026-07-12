import os
from argon2 import low_level
from cryptography.hazmat.primitives.ciphers.aead import AESGCMSIV

# A shared integer key used to derive the actual encryption key
SHARED_SECRET_INT = 987654321  

def keygen(secret: bytes, salt: bytes) -> bytes:
    return low_level.hash_secret_raw(
        secret=secret,
        salt=salt,
        time_cost=3,
        memory_cost=65536,  # 64 MB
        parallelism=4,
        hash_len=32,
        type=low_level.Type.ID,
    )

def buildkey(skey: int) -> bytes:
    secret = str(skey).encode("utf-8")
    salt = b"fixed-shared-salt-16b"[:16].ljust(16, b"0")  # exactly 16 bytes
    return keygen(secret, salt)

def encrypt_bytes(raw_bytes: bytes, key: bytes) -> bytes:
    nonce = os.urandom(12)
    # AES-GCM-SIV resists catastrophic leaks if nonce repeats
    ciphertext = AESGCMSIV(key).encrypt(nonce, raw_bytes, None)
    return nonce + ciphertext

def decrypt_bytes(encrypted_bytes: bytes, key: bytes) -> bytes:
    nonce = encrypted_bytes[:12]
    ciphertext = encrypted_bytes[12:]
    return AESGCMSIV(key).decrypt(nonce, ciphertext, None)

# Pre-compute the key once on import for performance
DERIVED_KEY = buildkey(SHARED_SECRET_INT)
