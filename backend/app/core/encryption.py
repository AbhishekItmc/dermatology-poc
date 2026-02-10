"""
Data encryption utilities

Requirements: 13.1, 13.5
"""
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import os
from typing import Union

from app.core.config import settings


def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derive encryption key from password using PBKDF2
    
    Requirements: 13.1
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


class DataEncryption:
    """
    Data encryption/decryption service
    
    Requirements: 13.1, 13.5
    """
    
    def __init__(self, key: Union[str, bytes] = None):
        """
        Initialize encryption service
        
        Args:
            key: Encryption key (uses settings.ENCRYPTION_KEY if not provided)
        """
        if key is None:
            key = settings.ENCRYPTION_KEY
        
        if isinstance(key, str):
            # Ensure key is 32 bytes for Fernet
            if len(key) < 32:
                key = key.ljust(32, '0')
            elif len(key) > 32:
                key = key[:32]
            
            # Derive key from password
            salt = b'dermatology_poc_salt'  # In production, use random salt per data
            key = derive_key(key, salt)
        
        self.cipher = Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data
        
        Requirements: 13.1
        
        Args:
            data: Data to encrypt (string or bytes)
        
        Returns:
            Encrypted data as bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data
        
        Requirements: 13.1
        
        Args:
            encrypted_data: Encrypted data as bytes
        
        Returns:
            Decrypted data as bytes
        """
        return self.cipher.decrypt(encrypted_data)
    
    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """
        Encrypt a file
        
        Requirements: 13.1
        
        Args:
            file_path: Path to file to encrypt
            output_path: Path to save encrypted file (defaults to file_path + '.enc')
        
        Returns:
            Path to encrypted file
        """
        if output_path is None:
            output_path = file_path + '.enc'
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        return output_path
    
    def decrypt_file(self, encrypted_file_path: str, output_path: str = None) -> str:
        """
        Decrypt a file
        
        Requirements: 13.1
        
        Args:
            encrypted_file_path: Path to encrypted file
            output_path: Path to save decrypted file
        
        Returns:
            Path to decrypted file
        """
        if output_path is None:
            if encrypted_file_path.endswith('.enc'):
                output_path = encrypted_file_path[:-4]
            else:
                output_path = encrypted_file_path + '.dec'
        
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_path
    
    def secure_delete(self, file_path: str, passes: int = 3) -> bool:
        """
        Securely delete a file by overwriting with random data
        
        Requirements: 13.5
        
        Args:
            file_path: Path to file to delete
            passes: Number of overwrite passes (default 3)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Overwrite file with random data multiple times
            for _ in range(passes):
                with open(file_path, 'wb') as f:
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Finally delete the file
            os.remove(file_path)
            
            return True
            
        except Exception as e:
            print(f"Error securely deleting file: {e}")
            return False


# Global encryption instance
encryption = DataEncryption()


def encrypt_data(data: Union[str, bytes]) -> bytes:
    """Convenience function to encrypt data"""
    return encryption.encrypt(data)


def decrypt_data(encrypted_data: bytes) -> bytes:
    """Convenience function to decrypt data"""
    return encryption.decrypt(encrypted_data)


def secure_delete_file(file_path: str, passes: int = 3) -> bool:
    """Convenience function to securely delete a file"""
    return encryption.secure_delete(file_path, passes)
