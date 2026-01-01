"""
Authentication modülü için temel testler.

Not: Bu testler MongoDB bağlantısı gerektirmez, sadece modüllerin
doğru çalıştığını ve import edildiğini test eder.
"""

import pytest
import sys
from pathlib import Path

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import_models():
    """Pydantic modellerinin import edildiğini test eder."""
    try:
        from api.models import (
            UserBase, UserCreate, UserInDB, UserResponse,
            Token, TokenData, PromptLogCreate
        )
        assert True, "Models imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import models: {e}")


def test_import_auth_utils():
    """Auth utils fonksiyonlarının import edildiğini test eder."""
    try:
        from api.auth_utils import (
            verify_password,
            get_password_hash,
            create_access_token,
            decode_access_token
        )
        assert True, "Auth utils imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import auth utils: {e}")


def test_password_hashing():
    """Password hashing ve verification test eder."""
    from api.auth_utils import get_password_hash, verify_password
    
    password = "TestPassword123"
    hashed = get_password_hash(password)
    
    # Hash farklı olmalı
    assert hashed != password, "Password should be hashed"
    
    # Doğrulama başarılı olmalı
    assert verify_password(password, hashed), "Password verification should succeed"
    
    # Yanlış password doğrulama başarısız olmalı
    assert not verify_password("WrongPassword", hashed), "Wrong password should fail"


def test_jwt_token_creation():
    """JWT token oluşturma ve decode test eder."""
    from api.auth_utils import create_access_token, decode_access_token
    from datetime import timedelta
    
    # Token oluştur
    test_data = {"sub": "test_user"}
    token = create_access_token(data=test_data, expires_delta=timedelta(minutes=15))
    
    assert isinstance(token, str), "Token should be a string"
    assert len(token) > 50, "Token should be reasonably long"
    
    # Token'ı decode et
    payload = decode_access_token(token)
    
    assert payload is not None, "Payload should not be None"
    assert payload.get("sub") == "test_user", "Username should match"
    assert "exp" in payload, "Token should have expiration"


def test_user_model_validation():
    """UserCreate model validation test eder."""
    from api.models import UserCreate
    from pydantic import ValidationError
    
    # Geçerli kullanıcı
    valid_user = UserCreate(
        username="test_user",
        email="test@example.com",
        password="SecurePass123",
        full_name="Test User",
        role="user"
    )
    
    assert valid_user.username == "test_user"
    assert valid_user.email == "test@example.com"
    
    # Geçersiz email
    with pytest.raises(ValidationError):
        UserCreate(
            username="test",
            email="invalid-email",  # Geçersiz format
            password="SecurePass123",
            full_name="Test"
        )
    
    # Çok kısa password
    with pytest.raises(ValidationError):
        UserCreate(
            username="test",
            email="test@example.com",
            password="short",  # Çok kısa
            full_name="Test"
        )


def test_token_model():
    """Token model test eder."""
    from api.models import Token
    
    token = Token(access_token="test_token_123")
    
    assert token.access_token == "test_token_123"
    assert token.token_type == "bearer"


def test_prompt_log_model():
    """PromptLogCreate model test eder."""
    from api.models import PromptLogCreate
    from pydantic import ValidationError
    
    # Geçerli log
    log = PromptLogCreate(
        text="This is a test review",
        sentiment="positive",
        confidence=0.95,
        prediction_time_ms=25.5
    )
    
    assert log.text == "This is a test review"
    assert log.sentiment == "positive"
    assert 0.0 <= log.confidence <= 1.0
    
    # Geçersiz confidence (>1)
    with pytest.raises(ValidationError):
        PromptLogCreate(
            text="Test",
            sentiment="positive",
            confidence=1.5,  # >1 geçersiz
            prediction_time_ms=20
        )


def test_import_crud():
    """CRUD operations'ın import edildiğini test eder."""
    try:
        from api.crud import (
            create_user,
            get_user_by_username,
            create_prompt_log
        )
        assert True, "CRUD operations imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import CRUD operations: {e}")


def test_import_dependencies():
    """Dependencies'in import edildiğini test eder."""
    try:
        from api.dependencies import (
            get_current_user,
            get_current_admin_user,
            oauth2_scheme
        )
        assert True, "Dependencies imported successfully"
    except ImportError as e:
        pytest.fail(f"Failed to import dependencies: {e}")


def test_import_auth_router():
    """Auth router'ın import edildiğini test eder."""
    try:
        from api.auth import router
        assert router is not None, "Router should be imported"
        assert hasattr(router, 'routes'), "Router should have routes"
    except ImportError as e:
        pytest.fail(f"Failed to import auth router: {e}")


def test_password_complexity():
    """Password complexity validation test eder."""
    from api.models import UserCreate
    from pydantic import ValidationError
    
    # Sadece harf - geçersiz
    with pytest.raises(ValidationError):
        UserCreate(
            username="test",
            email="test@example.com",
            password="onlyletters",  # Rakam yok
            full_name="Test"
        )
    
    # Sadece rakam - geçersiz
    with pytest.raises(ValidationError):
        UserCreate(
            username="test",
            email="test@example.com",
            password="12345678",  # Harf yok
            full_name="Test"
        )


def test_username_validation():
    """Username validation test eder."""
    from api.models import UserCreate
    from pydantic import ValidationError
    
    # Özel karakter içeren username - geçersiz
    with pytest.raises(ValidationError):
        UserCreate(
            username="test@user",  # @ geçersiz
            email="test@example.com",
            password="SecurePass123",
            full_name="Test"
        )
    
    # Geçerli username (underscore OK)
    valid_user = UserCreate(
        username="test_user_123",
        email="test@example.com",
        password="SecurePass123",
        full_name="Test"
    )
    
    assert valid_user.username == "test_user_123"


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Authentication Basic Tests")
    print("=" * 60)
    
    pytest.main([__file__, "-v", "--tb=short"])

