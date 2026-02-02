#!/usr/bin/env python3
"""
Rate Limiting Module for Stock Analyzer API

Implements per-IP and per-user rate limiting with sliding window algorithm.
Tracks request counts in memory with automatic cleanup of expired entries.
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from functools import wraps

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    # Per-IP limits (unauthenticated requests)
    ip_requests_per_minute: int = 30
    ip_requests_per_hour: int = 200
    ip_requests_per_day: int = 1000
    
    # Per-user limits (authenticated with API key)
    user_requests_per_minute: int = 60
    user_requests_per_hour: int = 500
    user_requests_per_day: int = 5000
    
    # Endpoint-specific limits (requests per minute)
    endpoint_limits: Dict[str, int] = field(default_factory=lambda: {
        "/api/analyze": 10,      # Heavy computation
        "/api/screener": 5,      # Very heavy computation
        "/api/symbols": 30,      # Light endpoint
        "/api/health": 100,      # Health checks
    })
    
    # Cleanup interval (seconds)
    cleanup_interval: int = 300  # 5 minutes


# Default configuration
DEFAULT_CONFIG = RateLimitConfig()


# =============================================================================
# Sliding Window Rate Limiter
# =============================================================================

class SlidingWindowRateLimiter:
    """
    Thread-safe sliding window rate limiter.
    
    Tracks requests in a sliding time window for accurate rate limiting.
    Supports multiple time windows (minute, hour, day) simultaneously.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or DEFAULT_CONFIG
        
        # Request timestamps: {identifier: [timestamp1, timestamp2, ...]}
        self._ip_requests: Dict[str, list] = defaultdict(list)
        self._user_requests: Dict[str, list] = defaultdict(list)
        self._endpoint_requests: Dict[str, list] = defaultdict(list)
        
        # Locks for thread safety
        self._ip_lock = threading.Lock()
        self._user_lock = threading.Lock()
        self._endpoint_lock = threading.Lock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread to clean up old entries."""
        def cleanup_loop():
            while True:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_old_entries()
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def _cleanup_old_entries(self):
        """Remove entries older than 24 hours."""
        cutoff = time.time() - 86400  # 24 hours
        
        with self._ip_lock:
            for key in list(self._ip_requests.keys()):
                self._ip_requests[key] = [
                    ts for ts in self._ip_requests[key] if ts > cutoff
                ]
                if not self._ip_requests[key]:
                    del self._ip_requests[key]
        
        with self._user_lock:
            for key in list(self._user_requests.keys()):
                self._user_requests[key] = [
                    ts for ts in self._user_requests[key] if ts > cutoff
                ]
                if not self._user_requests[key]:
                    del self._user_requests[key]
        
        with self._endpoint_lock:
            for key in list(self._endpoint_requests.keys()):
                self._endpoint_requests[key] = [
                    ts for ts in self._endpoint_requests[key] if ts > cutoff
                ]
                if not self._endpoint_requests[key]:
                    del self._endpoint_requests[key]
    
    def _count_requests_in_window(self, timestamps: list, window_seconds: int) -> int:
        """Count requests within the time window."""
        cutoff = time.time() - window_seconds
        return sum(1 for ts in timestamps if ts > cutoff)
    
    def _check_limits(
        self,
        timestamps: list,
        limits: Dict[int, int]  # {window_seconds: max_requests}
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Check if any limit is exceeded.
        
        Returns:
            (allowed, limit_type, retry_after_seconds)
        """
        now = time.time()
        
        for window_seconds, max_requests in limits.items():
            count = self._count_requests_in_window(timestamps, window_seconds)
            if count >= max_requests:
                # Calculate retry-after (when oldest request in window expires)
                cutoff = now - window_seconds
                window_requests = sorted([ts for ts in timestamps if ts > cutoff])
                if window_requests:
                    retry_after = int(window_requests[0] + window_seconds - now) + 1
                else:
                    retry_after = 1
                
                limit_type = self._window_to_name(window_seconds)
                return False, limit_type, retry_after
        
        return True, None, None
    
    def _window_to_name(self, seconds: int) -> str:
        """Convert window seconds to human-readable name."""
        if seconds <= 60:
            return "per-minute"
        elif seconds <= 3600:
            return "per-hour"
        else:
            return "per-day"
    
    def check_ip_limit(self, ip: str) -> Tuple[bool, Optional[str], Optional[int], Dict]:
        """
        Check if IP is within rate limits.
        
        Returns:
            (allowed, limit_type, retry_after, rate_info)
        """
        with self._ip_lock:
            timestamps = self._ip_requests[ip]
            
            limits = {
                60: self.config.ip_requests_per_minute,
                3600: self.config.ip_requests_per_hour,
                86400: self.config.ip_requests_per_day,
            }
            
            allowed, limit_type, retry_after = self._check_limits(timestamps, limits)
            
            # Build rate info for headers
            rate_info = {
                "minute": {
                    "limit": self.config.ip_requests_per_minute,
                    "remaining": max(0, self.config.ip_requests_per_minute - 
                                    self._count_requests_in_window(timestamps, 60)),
                    "reset": int(time.time()) + 60,
                },
                "hour": {
                    "limit": self.config.ip_requests_per_hour,
                    "remaining": max(0, self.config.ip_requests_per_hour - 
                                    self._count_requests_in_window(timestamps, 3600)),
                    "reset": int(time.time()) + 3600,
                },
            }
            
            return allowed, limit_type, retry_after, rate_info
    
    def check_user_limit(self, user_id: str) -> Tuple[bool, Optional[str], Optional[int], Dict]:
        """
        Check if user is within rate limits.
        
        Returns:
            (allowed, limit_type, retry_after, rate_info)
        """
        with self._user_lock:
            timestamps = self._user_requests[user_id]
            
            limits = {
                60: self.config.user_requests_per_minute,
                3600: self.config.user_requests_per_hour,
                86400: self.config.user_requests_per_day,
            }
            
            allowed, limit_type, retry_after = self._check_limits(timestamps, limits)
            
            rate_info = {
                "minute": {
                    "limit": self.config.user_requests_per_minute,
                    "remaining": max(0, self.config.user_requests_per_minute - 
                                    self._count_requests_in_window(timestamps, 60)),
                    "reset": int(time.time()) + 60,
                },
                "hour": {
                    "limit": self.config.user_requests_per_hour,
                    "remaining": max(0, self.config.user_requests_per_hour - 
                                    self._count_requests_in_window(timestamps, 3600)),
                    "reset": int(time.time()) + 3600,
                },
            }
            
            return allowed, limit_type, retry_after, rate_info
    
    def check_endpoint_limit(self, ip: str, endpoint: str) -> Tuple[bool, Optional[int]]:
        """
        Check endpoint-specific rate limit.
        
        Returns:
            (allowed, retry_after)
        """
        limit = self.config.endpoint_limits.get(endpoint)
        if limit is None:
            return True, None
        
        key = f"{ip}:{endpoint}"
        
        with self._endpoint_lock:
            timestamps = self._endpoint_requests[key]
            count = self._count_requests_in_window(timestamps, 60)
            
            if count >= limit:
                cutoff = time.time() - 60
                window_requests = sorted([ts for ts in timestamps if ts > cutoff])
                if window_requests:
                    retry_after = int(window_requests[0] + 60 - time.time()) + 1
                else:
                    retry_after = 1
                return False, retry_after
            
            return True, None
    
    def record_ip_request(self, ip: str):
        """Record a request from an IP."""
        with self._ip_lock:
            self._ip_requests[ip].append(time.time())
    
    def record_user_request(self, user_id: str):
        """Record a request from a user."""
        with self._user_lock:
            self._user_requests[user_id].append(time.time())
    
    def record_endpoint_request(self, ip: str, endpoint: str):
        """Record a request to an endpoint."""
        key = f"{ip}:{endpoint}"
        with self._endpoint_lock:
            self._endpoint_requests[key].append(time.time())
    
    def get_stats(self, include_sensitive: bool = False) -> Dict:
        """
        Get current rate limiter statistics.
        
        Args:
            include_sensitive: If True, include detailed counts (admin only).
                             Follows principle of least privilege.
        """
        with self._ip_lock:
            ip_count = len(self._ip_requests)
            ip_total = sum(len(v) for v in self._ip_requests.values())
        
        with self._user_lock:
            user_count = len(self._user_requests)
            user_total = sum(len(v) for v in self._user_requests.values())
        
        with self._endpoint_lock:
            endpoint_count = len(self._endpoint_requests)
        
        # Basic stats - safe to expose
        stats = {
            "active_sessions": ip_count + user_count,
        }
        
        # Detailed stats - only for admins (least privilege)
        if include_sensitive:
            stats.update({
                "tracked_ips": ip_count,
                "tracked_users": user_count,
                "tracked_endpoints": endpoint_count,
                "total_ip_requests": ip_total,
                "total_user_requests": user_total,
            })
        
        return stats


# Global rate limiter instance
rate_limiter = SlidingWindowRateLimiter()


# =============================================================================
# FastAPI Middleware & Dependencies
# =============================================================================

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header (for reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(",")[0].strip()
    
    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fall back to direct client IP
    if request.client:
        return request.client.host
    
    return "unknown"


def get_api_key(request: Request) -> Optional[str]:
    """Extract API key from request headers."""
    # Check Authorization header (Bearer token)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    
    # Check X-API-Key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key
    
    # Check query parameter (less secure, but convenient for testing)
    api_key = request.query_params.get("api_key")
    if api_key:
        return api_key
    
    return None


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware for rate limiting.
    
    Checks per-IP, per-user (if authenticated), and per-endpoint limits.
    Adds rate limit headers to response.
    """
    client_ip = get_client_ip(request)
    api_key = get_api_key(request)
    endpoint = request.url.path
    
    # Skip rate limiting for certain paths
    if endpoint in ["/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    # Check endpoint-specific limit first
    endpoint_allowed, endpoint_retry = rate_limiter.check_endpoint_limit(client_ip, endpoint)
    if not endpoint_allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Endpoint rate limit exceeded for {endpoint}",
                "retry_after": endpoint_retry,
            },
            headers={
                "Retry-After": str(endpoint_retry),
                "X-RateLimit-Type": "endpoint",
            }
        )
    
    # Check user limit if authenticated
    if api_key:
        user_allowed, user_limit_type, user_retry, user_rate_info = rate_limiter.check_user_limit(api_key)
        if not user_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"User rate limit exceeded ({user_limit_type})",
                    "retry_after": user_retry,
                },
                headers={
                    "Retry-After": str(user_retry),
                    "X-RateLimit-Limit": str(user_rate_info["minute"]["limit"]),
                    "X-RateLimit-Remaining": str(user_rate_info["minute"]["remaining"]),
                    "X-RateLimit-Reset": str(user_rate_info["minute"]["reset"]),
                    "X-RateLimit-Type": "user",
                }
            )
        rate_info = user_rate_info
    else:
        # Check IP limit for unauthenticated requests
        ip_allowed, ip_limit_type, ip_retry, ip_rate_info = rate_limiter.check_ip_limit(client_ip)
        if not ip_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"IP rate limit exceeded ({ip_limit_type})",
                    "retry_after": ip_retry,
                },
                headers={
                    "Retry-After": str(ip_retry),
                    "X-RateLimit-Limit": str(ip_rate_info["minute"]["limit"]),
                    "X-RateLimit-Remaining": str(ip_rate_info["minute"]["remaining"]),
                    "X-RateLimit-Reset": str(ip_rate_info["minute"]["reset"]),
                    "X-RateLimit-Type": "ip",
                }
            )
        rate_info = ip_rate_info
    
    # Record the request
    rate_limiter.record_ip_request(client_ip)
    rate_limiter.record_endpoint_request(client_ip, endpoint)
    if api_key:
        rate_limiter.record_user_request(api_key)
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers to response
    response.headers["X-RateLimit-Limit"] = str(rate_info["minute"]["limit"])
    response.headers["X-RateLimit-Remaining"] = str(max(0, rate_info["minute"]["remaining"] - 1))
    response.headers["X-RateLimit-Reset"] = str(rate_info["minute"]["reset"])
    
    return response


# =============================================================================
# Dependency for Route-Level Rate Limiting
# =============================================================================

def create_rate_limit_dependency(
    requests_per_minute: int = 30,
    requests_per_hour: int = 200
):
    """
    Create a FastAPI dependency for custom rate limiting on specific routes.
    
    Usage:
        @app.get("/api/heavy-endpoint", dependencies=[Depends(create_rate_limit_dependency(5, 50))])
        async def heavy_endpoint():
            ...
    """
    async def rate_limit_dependency(request: Request):
        client_ip = get_client_ip(request)
        api_key = get_api_key(request)
        
        # Use a unique key for this specific limit
        limit_key = f"custom:{request.url.path}:{client_ip}"
        if api_key:
            limit_key = f"custom:{request.url.path}:{api_key}"
        
        # Simple check (could be enhanced)
        allowed, _, retry_after, _ = rate_limiter.check_ip_limit(client_ip)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
    
    return rate_limit_dependency
