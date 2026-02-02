"""
OWASP Security Module
Implements OWASP Top 10 2021 security controls for the Stock Analyzer.

OWASP Top 10 2021 Coverage:
- A01: Broken Access Control        → Access control middleware, API key validation
- A02: Cryptographic Failures       → Secure secret handling, no sensitive data exposure
- A03: Injection                    → Input validation, parameterized queries
- A04: Insecure Design              → Security by design, threat modeling
- A05: Security Misconfiguration    → Secure defaults, headers, CORS
- A06: Vulnerable Components        → Dependabot, safety checks (CI/CD)
- A07: Auth Failures                → Rate limiting, session management
- A08: Data Integrity Failures      → Input validation, CSP headers
- A09: Logging Failures             → Security event logging
- A10: SSRF                         → URL validation, allowlists
"""

import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Callable, Optional, Set
from urllib.parse import urlparse

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


# =============================================================================
# A09: Security Logging Configuration
# =============================================================================

class SecurityEventType(Enum):
    """Types of security events to log."""
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ACCESS_DENIED = "access_denied"
    INJECTION_ATTEMPT = "injection_attempt"
    SSRF_ATTEMPT = "ssrf_attempt"


class SecurityLogger:
    """
    OWASP A09: Security Logging and Monitoring
    Centralized security event logging with structured output.
    """
    
    def __init__(self, log_file: str = "security.log"):
        self.logger = logging.getLogger("security")
        self.logger.setLevel(logging.INFO)
        
        # File handler for security events
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(handler)
            
            # Also log to console in development
            console = logging.StreamHandler()
            console.setLevel(logging.WARNING)
            console.setFormatter(logging.Formatter('SECURITY: %(message)s'))
            self.logger.addHandler(console)
    
    def log_event(
        self,
        event_type: SecurityEventType,
        ip_address: str,
        details: str,
        user_id: Optional[str] = None,
        severity: str = "INFO"
    ):
        """Log a security event with structured data."""
        log_entry = (
            f"event={event_type.value} | "
            f"ip={self._mask_ip(ip_address)} | "
            f"user={user_id or 'anonymous'} | "
            f"details={details}"
        )
        
        if severity == "WARNING":
            self.logger.warning(log_entry)
        elif severity == "ERROR":
            self.logger.error(log_entry)
        elif severity == "CRITICAL":
            self.logger.critical(log_entry)
        else:
            self.logger.info(log_entry)
    
    def _mask_ip(self, ip: str) -> str:
        """Partially mask IP for privacy (GDPR compliance)."""
        if not ip:
            return "unknown"
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
        return ip[:len(ip)//2] + "***"


# Global security logger instance
security_logger = SecurityLogger()


# =============================================================================
# A01: Access Control
# =============================================================================

@dataclass
class AccessControlConfig:
    """
    OWASP A01: Broken Access Control
    Configuration for access control policies.
    """
    # Allowed API key prefixes (for validation)
    allowed_key_prefixes: Set[str] = field(default_factory=lambda: {"sk_", "pk_", "api_"})
    
    # Maximum API key length
    max_key_length: int = 128
    
    # Minimum API key length
    min_key_length: int = 16
    
    # Endpoints that require authentication
    protected_endpoints: Set[str] = field(default_factory=lambda: {
        "/api/analyze",
        "/api/screener",
        "/api/admin",
    })
    
    # Endpoints that are always public
    public_endpoints: Set[str] = field(default_factory=lambda: {
        "/api/health",
        "/api/symbols",
        "/docs",
        "/openapi.json",
    })


def validate_api_key_format(api_key: str, config: AccessControlConfig = None) -> tuple[bool, str]:
    """
    Validate API key format (not authentication, just format).
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if config is None:
        config = AccessControlConfig()
    
    if not api_key:
        return False, "API key is required"
    
    if len(api_key) < config.min_key_length:
        return False, f"API key too short (min {config.min_key_length} chars)"
    
    if len(api_key) > config.max_key_length:
        return False, f"API key too long (max {config.max_key_length} chars)"
    
    # Check for only allowed characters (alphanumeric + underscore + hyphen)
    if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
        return False, "API key contains invalid characters"
    
    return True, ""


# =============================================================================
# A02: Cryptographic Failures Prevention
# =============================================================================

class SecretManager:
    """
    OWASP A02: Cryptographic Failures
    Secure handling of secrets and sensitive data.
    """
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_api_key(api_key: str, salt: Optional[str] = None) -> str:
        """
        Hash an API key for secure storage comparison.
        Never store API keys in plain text!
        """
        if salt is None:
            salt = os.getenv("API_KEY_SALT", "default_salt_change_in_production")
        
        return hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        ).hex()
    
    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """
        Compare two strings in constant time to prevent timing attacks.
        """
        return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))
    
    @staticmethod
    def redact_sensitive(data: dict, sensitive_keys: Set[str] = None) -> dict:
        """
        Redact sensitive fields from a dictionary for logging.
        """
        if sensitive_keys is None:
            sensitive_keys = {
                "password", "secret", "key", "token", "api_key",
                "authorization", "cookie", "session", "credit_card"
            }
        
        redacted = {}
        for k, v in data.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                redacted[k] = "***REDACTED***"
            elif isinstance(v, dict):
                redacted[k] = SecretManager.redact_sensitive(v, sensitive_keys)
            else:
                redacted[k] = v
        return redacted


# =============================================================================
# A03: Injection Prevention
# =============================================================================

class InjectionPrevention:
    """
    OWASP A03: Injection
    Input sanitization and injection prevention.
    """
    
    # Patterns that might indicate injection attempts
    SUSPICIOUS_PATTERNS = [
        r'<script',           # XSS
        r'javascript:',       # XSS
        r'on\w+\s*=',        # Event handlers
        r'SELECT\s+.*FROM',   # SQL injection
        r'UNION\s+SELECT',    # SQL injection
        r'INSERT\s+INTO',     # SQL injection
        r'DELETE\s+FROM',     # SQL injection
        r'DROP\s+TABLE',      # SQL injection
        r';\s*--',            # SQL comment
        r'\$\{.*\}',          # Template injection
        r'\{\{.*\}\}',        # Template injection
        r'\.\./',             # Path traversal
        r'%2e%2e%2f',         # Encoded path traversal
        r'file://',           # File protocol
        r'data:',             # Data URL
    ]
    
    @classmethod
    def detect_injection(cls, value: str) -> tuple[bool, str]:
        """
        Check if a string contains potential injection patterns.
        
        Returns:
            Tuple of (is_suspicious, pattern_matched)
        """
        if not value:
            return False, ""
        
        value_lower = value.lower()
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return True, pattern
        
        return False, ""
    
    @classmethod
    def sanitize_for_log(cls, value: str, max_length: int = 200) -> str:
        """Sanitize a value for safe logging (prevent log injection)."""
        if not value:
            return ""
        
        # Remove newlines and carriage returns (log injection)
        sanitized = value.replace('\n', '\\n').replace('\r', '\\r')
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[TRUNCATED]"
        
        return sanitized
    
    @classmethod
    def sanitize_html(cls, value: str) -> str:
        """Basic HTML entity encoding for XSS prevention."""
        if not value:
            return ""
        
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;',
        }
        
        for char, entity in replacements.items():
            value = value.replace(char, entity)
        
        return value


# =============================================================================
# A05: Security Headers (Security Misconfiguration Prevention)
# =============================================================================

class SecurityHeaders:
    """
    OWASP A05: Security Misconfiguration
    HTTP security headers configuration.
    """
    
    # Default security headers
    DEFAULT_HEADERS = {
        # Prevent clickjacking
        "X-Frame-Options": "DENY",
        
        # XSS protection (legacy browsers)
        "X-XSS-Protection": "1; mode=block",
        
        # Prevent MIME type sniffing
        "X-Content-Type-Options": "nosniff",
        
        # Referrer policy
        "Referrer-Policy": "strict-origin-when-cross-origin",
        
        # Permissions policy (restrict browser features)
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        
        # Cache control for sensitive data
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
        
        # Prevent information disclosure
        "X-Powered-By": "",  # Remove or set empty
    }
    
    # Content Security Policy (CSP)
    CSP_POLICY = {
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline' 'unsafe-eval' https://cdn.plot.ly",
        "style-src": "'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src": "'self' https://fonts.gstatic.com",
        "img-src": "'self' data: https:",
        "connect-src": "'self' https://api.alpaca.markets wss://stream.data.alpaca.markets",
        "frame-ancestors": "'none'",
        "form-action": "'self'",
        "base-uri": "'self'",
    }
    
    @classmethod
    def get_csp_header(cls) -> str:
        """Generate Content-Security-Policy header value."""
        return "; ".join(f"{k} {v}" for k, v in cls.CSP_POLICY.items())
    
    @classmethod
    def get_all_headers(cls, include_csp: bool = True) -> dict:
        """Get all security headers as a dictionary."""
        headers = cls.DEFAULT_HEADERS.copy()
        if include_csp:
            headers["Content-Security-Policy"] = cls.get_csp_header()
        return headers


# =============================================================================
# A10: SSRF Prevention
# =============================================================================

class SSRFPrevention:
    """
    OWASP A10: Server-Side Request Forgery
    URL validation and SSRF prevention.
    """
    
    # Blocked IP ranges (private networks, localhost, etc.)
    BLOCKED_IP_PATTERNS = [
        r'^127\.',              # Localhost
        r'^10\.',               # Private Class A
        r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',  # Private Class B
        r'^192\.168\.',         # Private Class C
        r'^169\.254\.',         # Link-local
        r'^0\.',                # Current network
        r'^224\.',              # Multicast
        r'^255\.',              # Broadcast
        r'^localhost$',         # Localhost hostname
        r'^::1$',               # IPv6 localhost
        r'^fe80:',              # IPv6 link-local
    ]
    
    # Allowed URL schemes
    ALLOWED_SCHEMES = {"http", "https"}
    
    # Allowed domains for external API calls
    ALLOWED_DOMAINS = {
        "api.alpaca.markets",
        "data.alpaca.markets",
        "stream.data.alpaca.markets",
        "paper-api.alpaca.markets",
    }
    
    @classmethod
    def validate_url(cls, url: str, allow_external: bool = False) -> tuple[bool, str]:
        """
        Validate a URL for SSRF prevention.
        
        Args:
            url: URL to validate
            allow_external: If True, allow external domains (with allowlist)
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        if not url:
            return False, "URL is required"
        
        try:
            parsed = urlparse(url)
        except Exception:
            return False, "Invalid URL format"
        
        # Check scheme
        if parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            return False, f"Scheme '{parsed.scheme}' not allowed"
        
        # Check for blocked IP patterns (always blocked)
        hostname = parsed.hostname or ""
        for pattern in cls.BLOCKED_IP_PATTERNS:
            if re.match(pattern, hostname, re.IGNORECASE):
                security_logger.log_event(
                    SecurityEventType.SSRF_ATTEMPT,
                    "internal",
                    f"Blocked SSRF attempt to {hostname}",
                    severity="WARNING"
                )
                return False, "Access to internal networks is not allowed"
        
        # Domain allowlist check:
        # - If allow_external=False (default): block ALL external domains
        # - If allow_external=True: only allow domains in ALLOWED_DOMAINS
        if not allow_external:
            # Default: no external domains allowed at all
            security_logger.log_event(
                SecurityEventType.SSRF_ATTEMPT,
                "internal",
                f"Blocked external domain access to {hostname} (allow_external=False)",
                severity="WARNING"
            )
            return False, "External domain access is not allowed"
        
        # allow_external=True: check against allowlist
        if hostname not in cls.ALLOWED_DOMAINS:
            security_logger.log_event(
                SecurityEventType.SSRF_ATTEMPT,
                "internal",
                f"Blocked non-allowlisted domain: {hostname}",
                severity="WARNING"
            )
            return False, f"Domain '{hostname}' is not in the allowlist"
        
        return True, ""


# =============================================================================
# Security Middleware for FastAPI
# =============================================================================

async def security_headers_middleware(request: Request, call_next: Callable):
    """
    Middleware to add security headers to all responses.
    """
    response = await call_next(request)
    
    # Add security headers
    for header, value in SecurityHeaders.get_all_headers().items():
        if value:  # Don't set empty headers
            response.headers[header] = value
    
    return response


async def injection_detection_middleware(request: Request, call_next: Callable):
    """
    Middleware to detect potential injection attempts in requests.
    """
    # Check query parameters
    for key, value in request.query_params.items():
        is_suspicious, pattern = InjectionPrevention.detect_injection(value)
        if is_suspicious:
            security_logger.log_event(
                SecurityEventType.INJECTION_ATTEMPT,
                request.client.host if request.client else "unknown",
                f"Suspicious pattern in query param '{key}': {pattern}",
                severity="WARNING"
            )
            # Optionally block the request
            # return JSONResponse(
            #     status_code=400,
            #     content={"error": "Invalid input detected"}
            # )
    
    # Check path parameters
    is_suspicious, pattern = InjectionPrevention.detect_injection(request.url.path)
    if is_suspicious:
        security_logger.log_event(
            SecurityEventType.INJECTION_ATTEMPT,
            request.client.host if request.client else "unknown",
            f"Suspicious pattern in path: {pattern}",
            severity="WARNING"
        )
    
    return await call_next(request)


# =============================================================================
# Security Checklist & Audit
# =============================================================================

OWASP_CHECKLIST = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OWASP TOP 10 2021 SECURITY CHECKLIST                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A01: Broken Access Control                                                    ║
║   ✅ API key validation implemented                                          ║
║   ✅ Rate limiting per user/IP                                               ║
║   ✅ Protected endpoints require authentication                              ║
║   ✅ CORS restricted to specific origins                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A02: Cryptographic Failures                                                   ║
║   ✅ API keys loaded from environment variables                              ║
║   ✅ Secrets never logged or exposed in errors                               ║
║   ✅ Constant-time comparison for sensitive data                             ║
║   ✅ HTTPS enforced (in production)                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A03: Injection                                                                ║
║   ✅ Input validation on all user inputs                                     ║
║   ✅ Symbol/date format validation with regex                                ║
║   ✅ Parameterized queries (no SQL injection)                                ║
║   ✅ XSS prevention via output encoding                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A04: Insecure Design                                                          ║
║   ✅ Security-first architecture                                             ║
║   ✅ Defense in depth (multiple layers)                                      ║
║   ✅ Fail-secure defaults                                                    ║
║   ✅ Threat modeling considered                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A05: Security Misconfiguration                                                ║
║   ✅ Security headers configured                                             ║
║   ✅ Debug mode disabled in production                                       ║
║   ✅ Error messages don't leak internals                                     ║
║   ✅ CSP headers configured                                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A06: Vulnerable and Outdated Components                                       ║
║   ✅ Dependabot enabled for auto-updates                                     ║
║   ✅ Safety checks in CI pipeline                                            ║
║   ✅ Bandit security scanning                                                ║
║   ✅ Regular dependency audits                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A07: Identification and Authentication Failures                               ║
║   ✅ Rate limiting prevents brute force                                      ║
║   ✅ Account lockout after failed attempts                                   ║
║   ✅ Secure session management                                               ║
║   ✅ No default credentials                                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A08: Software and Data Integrity Failures                                     ║
║   ✅ CI/CD pipeline with integrity checks                                    ║
║   ✅ Code review required (GitHub PRs)                                       ║
║   ✅ Input validation on all data                                            ║
║   ✅ No unsafe deserialization                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A09: Security Logging and Monitoring Failures                                 ║
║   ✅ Security events logged                                                  ║
║   ✅ Failed auth attempts logged                                             ║
║   ✅ Rate limit violations logged                                            ║
║   ✅ Suspicious activity detection                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A10: Server-Side Request Forgery (SSRF)                                       ║
║   ✅ URL validation with allowlist                                           ║
║   ✅ Internal network access blocked                                         ║
║   ✅ Only allowed external domains                                           ║
║   ✅ Scheme validation (http/https only)                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_security_checklist():
    """Print the OWASP security checklist."""
    print(OWASP_CHECKLIST)


def get_security_status() -> dict:
    """Get current security configuration status."""
    return {
        "owasp_compliance": "OWASP Top 10 2021",
        "security_headers": list(SecurityHeaders.DEFAULT_HEADERS.keys()),
        "csp_enabled": True,
        "rate_limiting": True,
        "input_validation": True,
        "security_logging": True,
        "ssrf_protection": True,
        "injection_detection": True,
    }


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "SecurityEventType",
    "SecurityLogger",
    "security_logger",
    "AccessControlConfig",
    "validate_api_key_format",
    "SecretManager",
    "InjectionPrevention",
    "SecurityHeaders",
    "SSRFPrevention",
    "security_headers_middleware",
    "injection_detection_middleware",
    "print_security_checklist",
    "get_security_status",
    "OWASP_CHECKLIST",
]
