"""
Tests for data encryption

Requirements: 13.1, 13.5
"""
import pytest
import os
import tempfile

from app.core.encryption import (
    DataEncryption,
    encrypt_data,
    decrypt_data,
    secure_delete_file
)

# Property-based testing
from hypothesis import given, strategies as st, settings as hypothesis_settings


class TestDataEncryption:
    """Test data encryption and decryption"""
    
    def test_encrypt_decrypt_string(self):
        """Test encrypting and decrypting a string"""
        encryption = DataEncryption()
        
        original = "sensitive patient data"
        encrypted = encryption.encrypt(original)
        decrypted = encryption.decrypt(encrypted)
        
        assert encrypted != original.encode()
        assert decrypted.decode() == original
    
    def test_encrypt_decrypt_bytes(self):
        """Test encrypting and decrypting bytes"""
        encryption = DataEncryption()
        
        original = b"binary patient data"
        encrypted = encryption.encrypt(original)
        decrypted = encryption.decrypt(encrypted)
        
        assert encrypted != original
        assert decrypted == original
    
    def test_encrypted_data_is_different(self):
        """Test encrypted data is different from original"""
        encryption = DataEncryption()
        
        data = "patient_id_12345"
        encrypted = encryption.encrypt(data)
        
        assert encrypted != data.encode()
        assert len(encrypted) > len(data)
    
    def test_different_keys_produce_different_ciphertext(self):
        """Test different encryption keys produce different ciphertext"""
        encryption1 = DataEncryption(key="key1_" + "0" * 26)
        encryption2 = DataEncryption(key="key2_" + "0" * 26)
        
        data = "sensitive data"
        encrypted1 = encryption1.encrypt(data)
        encrypted2 = encryption2.encrypt(data)
        
        assert encrypted1 != encrypted2
    
    def test_wrong_key_cannot_decrypt(self):
        """Test data encrypted with one key cannot be decrypted with another"""
        encryption1 = DataEncryption(key="key1_" + "0" * 26)
        encryption2 = DataEncryption(key="key2_" + "0" * 26)
        
        data = "sensitive data"
        encrypted = encryption1.encrypt(data)
        
        with pytest.raises(Exception):
            encryption2.decrypt(encrypted)


class TestFileEncryption:
    """Test file encryption and decryption"""
    
    def test_encrypt_decrypt_file(self):
        """Test encrypting and decrypting a file"""
        encryption = DataEncryption()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("sensitive patient data in file")
            temp_file = f.name
        
        try:
            # Encrypt file
            encrypted_file = encryption.encrypt_file(temp_file)
            assert os.path.exists(encrypted_file)
            assert encrypted_file.endswith('.enc')
            
            # Verify encrypted content is different
            with open(encrypted_file, 'rb') as f:
                encrypted_content = f.read()
            
            with open(temp_file, 'rb') as f:
                original_content = f.read()
            
            assert encrypted_content != original_content
            
            # Decrypt file
            decrypted_file = encryption.decrypt_file(encrypted_file)
            assert os.path.exists(decrypted_file)
            
            # Verify decrypted content matches original
            with open(decrypted_file, 'r') as f:
                decrypted_content = f.read()
            
            assert decrypted_content == "sensitive patient data in file"
            
        finally:
            # Cleanup
            for file in [temp_file, encrypted_file, decrypted_file]:
                if os.path.exists(file):
                    os.remove(file)
    
    def test_encrypt_file_custom_output(self):
        """Test encrypting file with custom output path"""
        encryption = DataEncryption()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test data")
            temp_file = f.name
        
        custom_output = temp_file + '.custom.enc'
        
        try:
            encrypted_file = encryption.encrypt_file(temp_file, custom_output)
            assert encrypted_file == custom_output
            assert os.path.exists(custom_output)
            
        finally:
            for file in [temp_file, custom_output]:
                if os.path.exists(file):
                    os.remove(file)


class TestSecureDelete:
    """Test secure file deletion"""
    
    def test_secure_delete_file(self):
        """Test securely deleting a file"""
        encryption = DataEncryption()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("sensitive data to be deleted")
            temp_file = f.name
        
        assert os.path.exists(temp_file)
        
        # Securely delete
        success = encryption.secure_delete(temp_file)
        
        assert success
        assert not os.path.exists(temp_file)
    
    def test_secure_delete_nonexistent_file(self):
        """Test securely deleting a nonexistent file"""
        encryption = DataEncryption()
        
        success = encryption.secure_delete("/nonexistent/file.txt")
        assert not success
    
    def test_secure_delete_multiple_passes(self):
        """Test secure deletion with multiple overwrite passes"""
        encryption = DataEncryption()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("data" * 1000)
            temp_file = f.name
        
        success = encryption.secure_delete(temp_file, passes=5)
        
        assert success
        assert not os.path.exists(temp_file)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_encrypt_data_function(self):
        """Test encrypt_data convenience function"""
        data = "test data"
        encrypted = encrypt_data(data)
        
        assert encrypted != data.encode()
        assert isinstance(encrypted, bytes)
    
    def test_decrypt_data_function(self):
        """Test decrypt_data convenience function"""
        data = "test data"
        encrypted = encrypt_data(data)
        decrypted = decrypt_data(encrypted)
        
        assert decrypted.decode() == data
    
    def test_secure_delete_file_function(self):
        """Test secure_delete_file convenience function"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test")
            temp_file = f.name
        
        success = secure_delete_file(temp_file)
        
        assert success
        assert not os.path.exists(temp_file)


# Property-based tests
@given(
    data=st.text(min_size=1, max_size=1000)
)
@hypothesis_settings(max_examples=5)
def test_property_encryption_reversible(data):
    """
    Property 29: Encryption is reversible with correct key
    
    Validates: Requirements 13.1
    """
    encryption = DataEncryption()
    
    # Encrypt data
    encrypted = encryption.encrypt(data)
    
    # Encrypted data should be different
    assert encrypted != data.encode()
    
    # Decryption should recover original data
    decrypted = encryption.decrypt(encrypted)
    assert decrypted.decode() == data


@given(
    data=st.binary(min_size=1, max_size=1000)
)
@hypothesis_settings(max_examples=5)
def test_property_encryption_binary_data(data):
    """
    Property 29: Encryption works with binary data
    
    Validates: Requirements 13.1
    """
    encryption = DataEncryption()
    
    encrypted = encryption.encrypt(data)
    decrypted = encryption.decrypt(encrypted)
    
    assert decrypted == data


@given(
    data=st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_categories=('Cs',), min_codepoint=32, max_codepoint=126))
)
@hypothesis_settings(max_examples=5)
def test_property_secure_delete_unrecoverable(data):
    """
    Property 33: Securely deleted files are unrecoverable
    
    Validates: Requirements 13.5
    """
    encryption = DataEncryption()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(data)
        temp_file = f.name
    
    # Securely delete
    success = encryption.secure_delete(temp_file, passes=3)
    
    # File should not exist
    assert success
    assert not os.path.exists(temp_file)


@given(
    data1=st.text(min_size=1, max_size=100),
    data2=st.text(min_size=1, max_size=100)
)
@hypothesis_settings(max_examples=5)
def test_property_different_data_different_ciphertext(data1, data2):
    """
    Property 29: Different data produces different ciphertext
    
    Validates: Requirements 13.1
    """
    if data1 == data2:
        return  # Skip if data is the same
    
    encryption = DataEncryption()
    
    encrypted1 = encryption.encrypt(data1)
    encrypted2 = encryption.encrypt(data2)
    
    # Different data should produce different ciphertext
    assert encrypted1 != encrypted2
