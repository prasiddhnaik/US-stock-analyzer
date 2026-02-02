#!/usr/bin/env python3
"""
Tests for Rate Limiter Module

Tests cover:
- Per-IP rate limiting
- Per-user rate limiting
- Endpoint-specific limits
- Sliding window accuracy
- Rate limit headers
"""

import time
import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rate_limiter import (
    SlidingWindowRateLimiter,
    RateLimitConfig,
    get_client_ip,
    get_api_key,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create a test configuration with low limits."""
    return RateLimitConfig(
        ip_requests_per_minute=5,
        ip_requests_per_hour=20,
        ip_requests_per_day=100,
        user_requests_per_minute=10,
        user_requests_per_hour=50,
        user_requests_per_day=200,
        endpoint_limits={
            "/api/test": 3,
            "/api/heavy": 1,
        },
        cleanup_interval=3600,  # Don't run cleanup during tests
    )


@pytest.fixture
def limiter(config):
    """Create a rate limiter with test config."""
    return SlidingWindowRateLimiter(config)


# =============================================================================
# IP Rate Limiting Tests
# =============================================================================

class TestIPRateLimiting:
    """Tests for per-IP rate limiting."""
    
    def test_allows_requests_under_limit(self, limiter):
        """Requests under the limit should be allowed."""
        ip = "192.168.1.1"
        
        for i in range(5):
            allowed, limit_type, retry_after, rate_info = limiter.check_ip_limit(ip)
            assert allowed is True, f"Request {i+1} should be allowed"
            limiter.record_ip_request(ip)
    
    def test_blocks_requests_over_minute_limit(self, limiter):
        """Requests over the minute limit should be blocked."""
        ip = "192.168.1.2"
        
        # Make 5 requests (the limit)
        for _ in range(5):
            limiter.record_ip_request(ip)
        
        # 6th request should be blocked
        allowed, limit_type, retry_after, rate_info = limiter.check_ip_limit(ip)
        assert allowed is False
        assert limit_type == "per-minute"
        assert retry_after is not None
        assert retry_after > 0
    
    def test_rate_info_shows_remaining(self, limiter):
        """Rate info should show correct remaining count."""
        ip = "192.168.1.3"
        
        # Check initial state
        allowed, _, _, rate_info = limiter.check_ip_limit(ip)
        assert rate_info["minute"]["remaining"] == 5
        
        # Make 2 requests
        limiter.record_ip_request(ip)
        limiter.record_ip_request(ip)
        
        # Check remaining
        allowed, _, _, rate_info = limiter.check_ip_limit(ip)
        assert rate_info["minute"]["remaining"] == 3
    
    def test_different_ips_have_separate_limits(self, limiter):
        """Different IPs should have independent limits."""
        ip1 = "192.168.1.10"
        ip2 = "192.168.1.11"
        
        # Exhaust ip1's limit
        for _ in range(5):
            limiter.record_ip_request(ip1)
        
        # ip1 should be blocked
        allowed1, _, _, _ = limiter.check_ip_limit(ip1)
        assert allowed1 is False
        
        # ip2 should still be allowed
        allowed2, _, _, _ = limiter.check_ip_limit(ip2)
        assert allowed2 is True


# =============================================================================
# User Rate Limiting Tests
# =============================================================================

class TestUserRateLimiting:
    """Tests for per-user (API key) rate limiting."""
    
    def test_allows_requests_under_limit(self, limiter):
        """User requests under limit should be allowed."""
        user_id = "api_key_12345"
        
        for i in range(10):
            allowed, limit_type, retry_after, rate_info = limiter.check_user_limit(user_id)
            assert allowed is True, f"Request {i+1} should be allowed"
            limiter.record_user_request(user_id)
    
    def test_blocks_requests_over_limit(self, limiter):
        """User requests over limit should be blocked."""
        user_id = "api_key_67890"
        
        # Make 10 requests (the limit)
        for _ in range(10):
            limiter.record_user_request(user_id)
        
        # 11th request should be blocked
        allowed, limit_type, retry_after, rate_info = limiter.check_user_limit(user_id)
        assert allowed is False
        assert limit_type == "per-minute"
    
    def test_users_have_higher_limits_than_ips(self, config):
        """Authenticated users should have higher limits."""
        assert config.user_requests_per_minute > config.ip_requests_per_minute
        assert config.user_requests_per_hour > config.ip_requests_per_hour
        assert config.user_requests_per_day > config.ip_requests_per_day


# =============================================================================
# Endpoint Rate Limiting Tests
# =============================================================================

class TestEndpointRateLimiting:
    """Tests for endpoint-specific rate limiting."""
    
    def test_endpoint_limit_applied(self, limiter):
        """Endpoint-specific limits should be applied."""
        ip = "192.168.1.20"
        endpoint = "/api/test"
        
        # Make 3 requests (the limit for this endpoint)
        for _ in range(3):
            limiter.record_endpoint_request(ip, endpoint)
        
        # 4th request should be blocked
        allowed, retry_after = limiter.check_endpoint_limit(ip, endpoint)
        assert allowed is False
        assert retry_after is not None
    
    def test_unknown_endpoint_no_limit(self, limiter):
        """Endpoints not in config should have no specific limit."""
        ip = "192.168.1.21"
        endpoint = "/api/unknown"
        
        # Should always be allowed (no endpoint-specific limit)
        allowed, retry_after = limiter.check_endpoint_limit(ip, endpoint)
        assert allowed is True
        assert retry_after is None
    
    def test_heavy_endpoint_has_strict_limit(self, limiter):
        """Heavy endpoints should have stricter limits."""
        ip = "192.168.1.22"
        endpoint = "/api/heavy"
        
        # Make 1 request (the limit)
        limiter.record_endpoint_request(ip, endpoint)
        
        # 2nd request should be blocked
        allowed, retry_after = limiter.check_endpoint_limit(ip, endpoint)
        assert allowed is False


# =============================================================================
# Sliding Window Tests
# =============================================================================

class TestSlidingWindow:
    """Tests for sliding window algorithm accuracy."""
    
    def test_old_requests_expire(self, limiter):
        """Old requests should not count against the limit."""
        ip = "192.168.1.30"
        
        # Manually add old timestamps (61 seconds ago)
        old_time = time.time() - 61
        limiter._ip_requests[ip] = [old_time] * 5
        
        # These old requests should not block new ones
        allowed, _, _, rate_info = limiter.check_ip_limit(ip)
        assert allowed is True
        assert rate_info["minute"]["remaining"] == 5
    
    def test_mixed_old_and_new_requests(self, limiter):
        """Mix of old and new requests should be counted correctly."""
        ip = "192.168.1.31"
        
        # Add 3 old requests (expired)
        old_time = time.time() - 61
        limiter._ip_requests[ip] = [old_time] * 3
        
        # Add 2 new requests
        limiter.record_ip_request(ip)
        limiter.record_ip_request(ip)
        
        # Should have 3 remaining (5 limit - 2 new requests)
        allowed, _, _, rate_info = limiter.check_ip_limit(ip)
        assert allowed is True
        assert rate_info["minute"]["remaining"] == 3


# =============================================================================
# Request Header Extraction Tests
# =============================================================================

class TestHeaderExtraction:
    """Tests for extracting client info from requests."""
    
    def test_get_client_ip_from_forwarded(self):
        """Should extract IP from X-Forwarded-For header."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "10.0.0.1, 10.0.0.2"}
        request.client = MagicMock(host="192.168.1.1")
        
        ip = get_client_ip(request)
        assert ip == "10.0.0.1"
    
    def test_get_client_ip_from_real_ip(self):
        """Should extract IP from X-Real-IP header."""
        request = MagicMock()
        request.headers = {"X-Real-IP": "10.0.0.5"}
        request.client = MagicMock(host="192.168.1.1")
        
        ip = get_client_ip(request)
        assert ip == "10.0.0.5"
    
    def test_get_client_ip_fallback(self):
        """Should fall back to direct client IP."""
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="192.168.1.100")
        
        ip = get_client_ip(request)
        assert ip == "192.168.1.100"
    
    def test_get_api_key_from_bearer(self):
        """Should extract API key from Bearer token."""
        request = MagicMock()
        request.headers = {"Authorization": "Bearer my_secret_key"}
        request.query_params = {}
        
        key = get_api_key(request)
        assert key == "my_secret_key"
    
    def test_get_api_key_from_header(self):
        """Should extract API key from X-API-Key header."""
        request = MagicMock()
        request.headers = {"X-API-Key": "header_key_123"}
        request.query_params = {}
        
        key = get_api_key(request)
        assert key == "header_key_123"
    
    def test_get_api_key_from_query(self):
        """Should extract API key from query parameter."""
        request = MagicMock()
        request.headers = {}
        request.query_params = {"api_key": "query_key_456"}
        
        key = get_api_key(request)
        assert key == "query_key_456"
    
    def test_get_api_key_none_when_missing(self):
        """Should return None when no API key provided."""
        request = MagicMock()
        request.headers = {}
        request.query_params = {}
        
        key = get_api_key(request)
        assert key is None


# =============================================================================
# Statistics Tests
# =============================================================================

class TestRateLimiterStats:
    """Tests for rate limiter statistics."""
    
    def test_stats_track_ips(self, limiter):
        """Stats should track number of IPs."""
        # Record requests from different IPs
        limiter.record_ip_request("192.168.1.50")
        limiter.record_ip_request("192.168.1.51")
        limiter.record_ip_request("192.168.1.52")
        
        # By default, sensitive stats are NOT included (security)
        stats = limiter.get_stats()
        assert "tracked_ips" not in stats
        
        # With include_sensitive=True, detailed stats are included
        stats_sensitive = limiter.get_stats(include_sensitive=True)
        assert stats_sensitive["tracked_ips"] == 3
    
    def test_stats_track_users(self, limiter):
        """Stats should track number of users (only with include_sensitive)."""
        limiter.record_user_request("user1")
        limiter.record_user_request("user2")
        
        # Sensitive stats require explicit opt-in
        stats = limiter.get_stats(include_sensitive=True)
        assert stats["tracked_users"] == 2
    
    def test_stats_track_total_requests(self, limiter):
        """Stats should track total request count (only with include_sensitive)."""
        ip = "192.168.1.60"
        limiter.record_ip_request(ip)
        limiter.record_ip_request(ip)
        limiter.record_ip_request(ip)
        
        # Sensitive stats require explicit opt-in
        stats = limiter.get_stats(include_sensitive=True)
        assert stats["total_ip_requests"] == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestRateLimiterIntegration:
    """Integration tests combining multiple features."""
    
    def test_ip_and_endpoint_limits_combined(self, limiter):
        """Both IP and endpoint limits should be checked."""
        ip = "192.168.1.70"
        endpoint = "/api/heavy"  # Limit of 1
        
        # First request should pass both checks
        ip_allowed, _, _, _ = limiter.check_ip_limit(ip)
        endpoint_allowed, _ = limiter.check_endpoint_limit(ip, endpoint)
        assert ip_allowed is True
        assert endpoint_allowed is True
        
        # Record the request
        limiter.record_ip_request(ip)
        limiter.record_endpoint_request(ip, endpoint)
        
        # Second request: IP should still be allowed, endpoint should be blocked
        ip_allowed, _, _, _ = limiter.check_ip_limit(ip)
        endpoint_allowed, _ = limiter.check_endpoint_limit(ip, endpoint)
        assert ip_allowed is True  # IP limit not reached
        assert endpoint_allowed is False  # Endpoint limit reached
    
    def test_authenticated_user_bypasses_ip_limit(self, limiter):
        """Authenticated users use user limits, not IP limits."""
        ip = "192.168.1.80"
        user_id = "premium_user"
        
        # Exhaust IP limit
        for _ in range(5):
            limiter.record_ip_request(ip)
        
        # IP limit should be reached
        ip_allowed, _, _, _ = limiter.check_ip_limit(ip)
        assert ip_allowed is False
        
        # But user limit should still have capacity
        user_allowed, _, _, _ = limiter.check_user_limit(user_id)
        assert user_allowed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
