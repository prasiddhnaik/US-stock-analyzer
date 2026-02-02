"""
Unit tests for security.py
Tests OWASP security controls implementation.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security import (
    SecurityEventType,
    SecurityLogger,
    AccessControlConfig,
    validate_api_key_format,
    SecretManager,
    InjectionPrevention,
    SecurityHeaders,
    SSRFPrevention,
    get_security_status,
)


# =============================================================================
# A01: Access Control Tests
# =============================================================================

class TestAccessControl:
    """Tests for access control validation."""
    
    def test_valid_api_key_format(self):
        """Test valid API key formats are accepted."""
        valid_keys = [
            "sk_test_1234567890abcdef",
            "api_key_production_xyz",
            "abcdefghijklmnop",
            "A1B2C3D4E5F6G7H8",
        ]
        for key in valid_keys:
            is_valid, error = validate_api_key_format(key)
            assert is_valid, f"Key '{key}' should be valid: {error}"
    
    def test_api_key_too_short(self):
        """Test short API keys are rejected."""
        is_valid, error = validate_api_key_format("short")
        assert not is_valid
        assert "too short" in error.lower()
    
    def test_api_key_too_long(self):
        """Test overly long API keys are rejected."""
        long_key = "a" * 200
        is_valid, error = validate_api_key_format(long_key)
        assert not is_valid
        assert "too long" in error.lower()
    
    def test_api_key_invalid_chars(self):
        """Test API keys with invalid characters are rejected."""
        invalid_keys = [
            "key with spaces",
            "key@special!chars",
            "key<script>",
            "key;DROP TABLE",
        ]
        for key in invalid_keys:
            is_valid, error = validate_api_key_format(key)
            assert not is_valid, f"Key '{key}' should be invalid"
    
    def test_empty_api_key(self):
        """Test empty API key is rejected."""
        is_valid, error = validate_api_key_format("")
        assert not is_valid
        assert "required" in error.lower()


# =============================================================================
# A02: Cryptographic Tests
# =============================================================================

class TestCryptography:
    """Tests for cryptographic functions."""
    
    def test_secure_token_generation(self):
        """Test secure token generation produces unique tokens."""
        tokens = [SecretManager.generate_secure_token() for _ in range(100)]
        unique_tokens = set(tokens)
        assert len(unique_tokens) == 100, "Tokens should be unique"
    
    def test_secure_token_length(self):
        """Test secure tokens have correct length."""
        token = SecretManager.generate_secure_token(32)
        assert len(token) >= 32, "Token should be at least 32 characters"
    
    def test_api_key_hashing(self):
        """Test API key hashing is deterministic."""
        key = "test_api_key_12345"
        hash1 = SecretManager.hash_api_key(key)
        hash2 = SecretManager.hash_api_key(key)
        assert hash1 == hash2, "Same key should produce same hash"
    
    def test_different_keys_different_hashes(self):
        """Test different keys produce different hashes."""
        hash1 = SecretManager.hash_api_key("key1_abcdefghijklmno")
        hash2 = SecretManager.hash_api_key("key2_abcdefghijklmno")
        assert hash1 != hash2, "Different keys should produce different hashes"
    
    def test_constant_time_compare(self):
        """Test constant time comparison."""
        assert SecretManager.constant_time_compare("abc", "abc")
        assert not SecretManager.constant_time_compare("abc", "abd")
    
    def test_redact_sensitive(self):
        """Test sensitive data redaction."""
        data = {
            "username": "john",
            "password": "secret123",
            "api_key": "sk_test_123",
            "safe_field": "visible",
        }
        redacted = SecretManager.redact_sensitive(data)
        assert redacted["username"] == "john"
        assert redacted["password"] == "***REDACTED***"
        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["safe_field"] == "visible"


# =============================================================================
# A03: Injection Prevention Tests
# =============================================================================

class TestInjectionPrevention:
    """Tests for injection attack detection."""
    
    def test_detect_xss_script_tag(self):
        """Test XSS script tag detection."""
        is_suspicious, pattern = InjectionPrevention.detect_injection("<script>alert(1)</script>")
        assert is_suspicious
        assert "script" in pattern.lower()
    
    def test_detect_xss_event_handler(self):
        """Test XSS event handler detection."""
        is_suspicious, _ = InjectionPrevention.detect_injection('onmouseover="alert(1)"')
        assert is_suspicious
    
    def test_detect_sql_injection(self):
        """Test SQL injection detection."""
        sql_payloads = [
            "SELECT * FROM users",
            "1; DROP TABLE users--",
            "UNION SELECT password FROM users",
        ]
        for payload in sql_payloads:
            is_suspicious, _ = InjectionPrevention.detect_injection(payload)
            assert is_suspicious, f"Should detect SQL injection: {payload}"
    
    def test_detect_path_traversal(self):
        """Test path traversal detection."""
        is_suspicious, _ = InjectionPrevention.detect_injection("../../../etc/passwd")
        assert is_suspicious
    
    def test_safe_input_passes(self):
        """Test safe inputs are not flagged."""
        safe_inputs = [
            "AAPL",
            "Microsoft Corporation",
            "2024-01-15",
            "Hello World",
            "price_target_50",
        ]
        for inp in safe_inputs:
            is_suspicious, _ = InjectionPrevention.detect_injection(inp)
            assert not is_suspicious, f"Safe input '{inp}' should not be flagged"
    
    def test_sanitize_for_log(self):
        """Test log sanitization removes newlines."""
        malicious = "Normal text\nINJECTED LOG ENTRY\r\nMore injection"
        sanitized = InjectionPrevention.sanitize_for_log(malicious)
        assert "\n" not in sanitized
        assert "\r" not in sanitized
        assert "\\n" in sanitized
    
    def test_sanitize_html(self):
        """Test HTML sanitization."""
        html = '<script>alert("xss")</script>'
        sanitized = InjectionPrevention.sanitize_html(html)
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "&lt;" in sanitized


# =============================================================================
# A05: Security Headers Tests
# =============================================================================

class TestSecurityHeaders:
    """Tests for security headers configuration."""
    
    def test_default_headers_present(self):
        """Test all recommended headers are configured."""
        headers = SecurityHeaders.DEFAULT_HEADERS
        required = [
            "X-Frame-Options",
            "X-XSS-Protection",
            "X-Content-Type-Options",
            "Referrer-Policy",
        ]
        for header in required:
            assert header in headers, f"Missing required header: {header}"
    
    def test_x_frame_options_deny(self):
        """Test X-Frame-Options is set to DENY."""
        headers = SecurityHeaders.DEFAULT_HEADERS
        assert headers["X-Frame-Options"] == "DENY"
    
    def test_csp_header_generated(self):
        """Test CSP header is properly generated."""
        csp = SecurityHeaders.get_csp_header()
        assert "default-src" in csp
        assert "'self'" in csp
    
    def test_get_all_headers(self):
        """Test get_all_headers returns complete set."""
        headers = SecurityHeaders.get_all_headers()
        assert len(headers) > 5  # Should have multiple headers
        assert "X-Frame-Options" in headers


# =============================================================================
# A10: SSRF Prevention Tests
# =============================================================================

class TestSSRFPrevention:
    """Tests for SSRF prevention."""
    
    def test_blocks_localhost(self):
        """Test localhost URLs are blocked."""
        is_safe, error = SSRFPrevention.validate_url("http://127.0.0.1/admin", allow_external=True)
        assert not is_safe
        assert "internal" in error.lower()
    
    def test_blocks_private_network(self):
        """Test private network IPs are blocked."""
        private_urls = [
            "http://10.0.0.1/internal",
            "http://192.168.1.1/admin",
            "http://172.16.0.1/secret",
        ]
        for url in private_urls:
            is_safe, _ = SSRFPrevention.validate_url(url, allow_external=True)
            assert not is_safe, f"Should block private URL: {url}"
    
    def test_blocks_file_protocol(self):
        """Test file:// protocol is blocked."""
        is_safe, error = SSRFPrevention.validate_url("file:///etc/passwd")
        assert not is_safe
        assert "scheme" in error.lower()
    
    def test_allows_https_to_allowlisted_domain(self):
        """Test HTTPS URLs to allowed domains pass when allow_external=True."""
        is_safe, _ = SSRFPrevention.validate_url(
            "https://api.alpaca.markets/v2/stocks",
            allow_external=True
        )
        assert is_safe
    
    def test_blocks_external_by_default(self):
        """Test external domains are blocked by default (allow_external=False)."""
        # Even allowlisted domains should be blocked when allow_external=False
        is_safe, error = SSRFPrevention.validate_url("https://api.alpaca.markets/v2/stocks")
        assert not is_safe
        assert "not allowed" in error.lower()
        
        # Non-allowlisted domains should also be blocked
        is_safe, error = SSRFPrevention.validate_url("https://evil.com/steal-data")
        assert not is_safe
        assert "not allowed" in error.lower()
    
    def test_blocks_non_allowlisted_domain(self):
        """Test non-allowlisted domains are blocked even with allow_external=True."""
        is_safe, error = SSRFPrevention.validate_url(
            "https://evil.com/steal-data",
            allow_external=True
        )
        assert not is_safe
        assert "allowlist" in error.lower()


# =============================================================================
# Security Status Tests
# =============================================================================

class TestSecurityStatus:
    """Tests for security status reporting."""
    
    def test_security_status_returns_dict(self):
        """Test security status returns proper structure."""
        status = get_security_status()
        assert isinstance(status, dict)
        assert "owasp_compliance" in status
    
    def test_security_features_enabled(self):
        """Test all security features are marked as enabled."""
        status = get_security_status()
        assert status["rate_limiting"] is True
        assert status["input_validation"] is True
        assert status["security_logging"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
