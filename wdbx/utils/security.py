"""
Security utilities for WDBX.

This module provides security-related functionality for WDBX.
"""

import os
import logging
import hashlib
import hmac
import base64
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

logger = logging.getLogger(__name__)


class WDBXSecurity:
    """
    Security manager for WDBX.

    This class provides encryption, authentication, and access control
    for WDBX data and API operations.

    Attributes:
        wdbx: Reference to the WDBX instance
        secret_key: Secret key for cryptographic operations
        token_expiry: Token expiry time in seconds
        access_policies: Access control policies
    """

    def __init__(
        self,
        wdbx=None,
        secret_key: Optional[str] = None,
        token_expiry: int = 86400,  # 24 hours
        enable_encryption: bool = False,
        enable_authentication: bool = False,
        enable_access_control: bool = False,
    ):
        """
        Initialize the security manager.

        Args:
            wdbx: Optional reference to the WDBX instance
            secret_key: Secret key for cryptographic operations (generated if None)
            token_expiry: Token expiry time in seconds
            enable_encryption: Whether to enable data encryption
            enable_authentication: Whether to enable authentication
            enable_access_control: Whether to enable access control
        """
        self.wdbx = wdbx
        self.secret_key = secret_key or self._generate_secret_key()
        self.token_expiry = token_expiry
        self.enable_encryption = enable_encryption
        self.enable_authentication = enable_authentication
        self.enable_access_control = enable_access_control

        # Active tokens
        self.active_tokens = {}

        # Access control policies
        self.access_policies = {
            "default": {
                "read": True,
                "write": False,
                "delete": False,
                "admin": False,
            }
        }

        logger.info(
            f"Initialized security manager with encryption={enable_encryption}, "
            f"authentication={enable_authentication}, access_control={enable_access_control}"
        )

    def _generate_secret_key(self) -> str:
        """
        Generate a random secret key.

        Returns:
            Random secret key as a base64-encoded string
        """
        key = os.urandom(32)
        return base64.b64encode(key).decode("utf-8")

    def hash_password(
        self, password: str, salt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Hash a password with a salt.

        Args:
            password: Password to hash
            salt: Optional salt (generated if None)

        Returns:
            Tuple of (hashed_password, salt)
        """
        if salt is None:
            salt = os.urandom(16).hex()

        # Use PBKDF2 with SHA-256
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000,  # 100,000 iterations
            dklen=32,
        ).hex()

        return key, salt

    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Password to verify
            hashed_password: Stored hash
            salt: Salt used for hashing

        Returns:
            True if the password matches, False otherwise
        """
        key, _ = self.hash_password(password, salt)
        return hmac.compare_digest(key, hashed_password)

    def generate_token(self, user_id: str, user_roles: List[str] = None) -> str:
        """
        Generate an authentication token.

        Args:
            user_id: User ID
            user_roles: List of user roles

        Returns:
            Authentication token
        """
        if not self.enable_authentication:
            logger.warning("Authentication is disabled, but generate_token was called")

        # Prepare token data
        token_data = {
            "user_id": user_id,
            "roles": user_roles or ["user"],
            "exp": int(time.time()) + self.token_expiry,
            "iat": int(time.time()),
            "jti": os.urandom(8).hex(),
        }

        # Encode token data
        token_json = json.dumps(token_data)
        token_bytes = token_json.encode("utf-8")

        # Sign token
        signature = hmac.new(
            self.secret_key.encode("utf-8"), token_bytes, hashlib.sha256
        ).digest()

        # Combine token and signature
        combined = base64.b64encode(token_bytes) + b"." + base64.b64encode(signature)
        token = combined.decode("utf-8")

        # Store active token
        self.active_tokens[token] = token_data

        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify an authentication token.

        Args:
            token: Authentication token

        Returns:
            Token data if valid, None otherwise
        """
        if not self.enable_authentication:
            logger.warning("Authentication is disabled, but verify_token was called")
            return {"user_id": "anonymous", "roles": ["user"]}

        try:
            # Check if token is in active tokens
            if token in self.active_tokens:
                token_data = self.active_tokens[token]

                # Check expiration
                if token_data["exp"] < int(time.time()):
                    # Token expired
                    del self.active_tokens[token]
                    return None

                return token_data

            # Split token and signature
            parts = token.split(".")
            if len(parts) != 2:
                return None

            token_b64, signature_b64 = parts

            # Decode token
            try:
                token_bytes = base64.b64decode(token_b64)
                token_data = json.loads(token_bytes.decode("utf-8"))
            except Exception:
                return None

            # Check expiration
            if token_data.get("exp", 0) < int(time.time()):
                return None

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode("utf-8"), token_bytes, hashlib.sha256
            ).digest()

            actual_signature = base64.b64decode(signature_b64)

            if not hmac.compare_digest(expected_signature, actual_signature):
                return None

            # Store active token
            self.active_tokens[token] = token_data

            return token_data
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None

    def revoke_token(self, token: str) -> bool:
        """
        Revoke an authentication token.

        Args:
            token: Authentication token

        Returns:
            True if the token was revoked, False otherwise
        """
        if token in self.active_tokens:
            del self.active_tokens[token]
            return True
        return False

    def revoke_all_tokens(self, user_id: Optional[str] = None) -> int:
        """
        Revoke all tokens for a user, or all tokens if user_id is None.

        Args:
            user_id: Optional user ID

        Returns:
            Number of tokens revoked
        """
        if user_id is None:
            # Revoke all tokens
            count = len(self.active_tokens)
            self.active_tokens.clear()
            return count

        # Revoke tokens for a specific user
        tokens_to_revoke = []
        for token, token_data in self.active_tokens.items():
            if token_data.get("user_id") == user_id:
                tokens_to_revoke.append(token)

        for token in tokens_to_revoke:
            del self.active_tokens[token]

        return len(tokens_to_revoke)

    def set_access_policy(self, role: str, permissions: Dict[str, bool]) -> None:
        """
        Set access policy for a role.

        Args:
            role: Role name
            permissions: Dictionary mapping permission names to boolean values
        """
        if not self.enable_access_control:
            logger.warning(
                "Access control is disabled, but set_access_policy was called"
            )

        self.access_policies[role] = permissions

    def check_permission(self, token: Optional[str], permission: str) -> bool:
        """
        Check if a token has a permission.

        Args:
            token: Authentication token (None for anonymous access)
            permission: Permission to check

        Returns:
            True if the token has the permission, False otherwise
        """
        if not self.enable_access_control:
            return True

        if not self.enable_authentication:
            # Use default policy if authentication is disabled
            return self.access_policies.get("default", {}).get(permission, False)

        if token is None:
            # Anonymous access
            return self.access_policies.get("anonymous", {}).get(permission, False)

        # Verify token
        token_data = self.verify_token(token)
        if token_data is None:
            return False

        # Check permissions for each role
        roles = token_data.get("roles", ["user"])

        # Check if user has admin permission in any role
        for role in roles:
            if self.access_policies.get(role, {}).get("admin", False):
                return True

        # Check specific permission
        for role in roles:
            if self.access_policies.get(role, {}).get(permission, False):
                return True

        return False

    def encrypt_data(self, data: Union[str, bytes, Dict, List]) -> str:
        """
        Encrypt data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data as a string

        Raises:
            ValueError: If encryption is disabled
            ImportError: If required dependencies are not installed
        """
        if not self.enable_encryption:
            raise ValueError("Encryption is disabled")

        try:
            from cryptography.fernet import Fernet

            # Convert data to JSON if it's a dictionary or list
            if isinstance(data, (dict, list)):
                data = json.dumps(data)

            # Convert data to bytes if it's a string
            if isinstance(data, str):
                data = data.encode("utf-8")

            # Create encryption key from secret key
            key = base64.urlsafe_b64encode(
                hashlib.sha256(self.secret_key.encode("utf-8")).digest()
            )
            fernet = Fernet(key)

            # Encrypt data
            encrypted = fernet.encrypt(data)

            return base64.b64encode(encrypted).decode("utf-8")
        except ImportError:
            logger.error("cryptography not installed, required for encryption")
            raise ImportError(
                "cryptography is required for encryption. Install with: pip install cryptography"
            )
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise ValueError(f"Error encrypting data: {e}")

    def decrypt_data(
        self, encrypted_data: str, data_type: str = "bytes"
    ) -> Union[str, bytes, Dict, List]:
        """
        Decrypt data.

        Args:
            encrypted_data: Encrypted data
            data_type: Type of data to return ('bytes', 'str', 'json')

        Returns:
            Decrypted data

        Raises:
            ValueError: If encryption is disabled or decryption fails
            ImportError: If required dependencies are not installed
        """
        if not self.enable_encryption:
            raise ValueError("Encryption is disabled")

        try:
            from cryptography.fernet import Fernet

            # Create encryption key from secret key
            key = base64.urlsafe_b64encode(
                hashlib.sha256(self.secret_key.encode("utf-8")).digest()
            )
            fernet = Fernet(key)

            # Decrypt data
            encrypted = base64.b64decode(encrypted_data)
            decrypted = fernet.decrypt(encrypted)

            # Convert to requested type
            if data_type == "bytes":
                return decrypted
            elif data_type == "str":
                return decrypted.decode("utf-8")
            elif data_type == "json":
                return json.loads(decrypted.decode("utf-8"))
            else:
                raise ValueError(f"Unsupported data_type: {data_type}")
        except ImportError:
            logger.error("cryptography not installed, required for encryption")
            raise ImportError(
                "cryptography is required for encryption. Install with: pip install cryptography"
            )
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise ValueError(f"Error decrypting data: {e}")

    def secure_metadata(
        self, metadata: Dict[str, Any], sensitive_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in metadata.

        Args:
            metadata: Metadata dictionary
            sensitive_fields: List of field names to encrypt

        Returns:
            Metadata with encrypted fields

        Raises:
            ValueError: If encryption is disabled
        """
        if not self.enable_encryption:
            raise ValueError("Encryption is disabled")

        secured_metadata = metadata.copy()

        for field in sensitive_fields:
            if field in secured_metadata:
                value = secured_metadata[field]

                # Encrypt the field
                encrypted_value = self.encrypt_data(value)

                # Replace with encrypted value
                secured_metadata[field] = {"_encrypted": encrypted_value}

        return secured_metadata

    def restore_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted fields in metadata.

        Args:
            metadata: Metadata dictionary with encrypted fields

        Returns:
            Metadata with decrypted fields

        Raises:
            ValueError: If encryption is disabled
        """
        if not self.enable_encryption:
            raise ValueError("Encryption is disabled")

        restored_metadata = metadata.copy()

        for field, value in metadata.items():
            if isinstance(value, dict) and "_encrypted" in value:
                encrypted_value = value["_encrypted"]

                # Decrypt the field
                try:
                    decrypted_value = self.decrypt_data(
                        encrypted_value, data_type="json"
                    )

                    # Replace with decrypted value
                    restored_metadata[field] = decrypted_value
                except Exception as e:
                    logger.error(f"Error decrypting field {field}: {e}")
                    # Keep the encrypted value

        return restored_metadata

    def create_auth_middleware(self) -> Callable:
        """
        Create authentication middleware for FastAPI.

        Returns:
            Middleware function

        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            from fastapi import Request, HTTPException, status
            from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

            security = HTTPBearer()

            async def auth_middleware(
                request: Request, credentials: HTTPAuthorizationCredentials = None
            ):
                if not self.enable_authentication:
                    return {"user_id": "anonymous", "roles": ["user"]}

                if credentials is None:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Missing authentication credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                token_data = self.verify_token(credentials.credentials)
                if token_data is None:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired token",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                return token_data

            return auth_middleware
        except ImportError:
            logger.error("fastapi not installed, required for auth middleware")
            raise ImportError(
                "fastapi is required for auth middleware. Install with: pip install fastapi"
            )
