#!/usr/bin/env python3
"""
Strict Input Validation & Sanitization Module

Implements OWASP-compliant input validation with:
- Schema-based validation (define allowed fields explicitly)
- Type checking (strict type enforcement)
- Length limits (prevent buffer overflow/DoS)
- Unexpected field rejection (prevent mass assignment)
- Deep sanitization (prevent injection attacks)

Usage:
    validator = StrictValidator(AnalyzeRequestSchema)
    clean_data = validator.validate(user_input)  # raises StrictValidationError
"""

import re
import html
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import json


# =============================================================================
# Validation Exceptions
# =============================================================================

class StrictValidationError(Exception):
    """Raised when validation fails."""
    
    def __init__(self, errors: List[Dict[str, Any]]):
        self.errors = errors
        message = f"Validation failed: {len(errors)} error(s)"
        super().__init__(message)
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "error": "Validation failed",
            "error_code": "VALIDATION_ERROR",
            "details": self.errors,
        }


# =============================================================================
# Field Types
# =============================================================================

class FieldType(Enum):
    """Supported field types for strict validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"           # YYYY-MM-DD
    DATETIME = "datetime"   # ISO 8601
    EMAIL = "email"
    SYMBOL = "symbol"       # Stock ticker
    LIST = "list"
    DICT = "dict"
    ENUM = "enum"


# =============================================================================
# Field Schema Definition
# =============================================================================

@dataclass
class FieldSchema:
    """
    Schema definition for a single field.
    
    Attributes:
        name: Field name
        field_type: Expected type
        required: Whether field is required
        default: Default value if not provided
        min_length: Minimum string length
        max_length: Maximum string length
        min_value: Minimum numeric value
        max_value: Maximum numeric value
        pattern: Regex pattern for string validation
        choices: Allowed values (for enum-like validation)
        item_schema: Schema for list items
        nested_schema: Schema for nested objects
        sanitizer: Custom sanitization function
        validator: Custom validation function
        description: Field description (for documentation)
    """
    name: str
    field_type: FieldType
    required: bool = False
    default: Any = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    choices: Optional[List[Any]] = None
    item_schema: Optional['FieldSchema'] = None
    nested_schema: Optional['RequestSchema'] = None
    sanitizer: Optional[Callable[[Any], Any]] = None
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""


@dataclass
class RequestSchema:
    """
    Schema definition for a request.
    
    Attributes:
        name: Schema name
        fields: Dict of field name -> FieldSchema
        allow_extra_fields: If False, reject unknown fields
        max_fields: Maximum number of fields allowed
        max_size_bytes: Maximum JSON size in bytes
    """
    name: str
    fields: Dict[str, FieldSchema] = field(default_factory=dict)
    allow_extra_fields: bool = False  # STRICT: reject unknown fields by default
    max_fields: int = 50
    max_size_bytes: int = 100_000  # 100KB default


# =============================================================================
# Sanitizers
# =============================================================================

class Sanitizers:
    """Collection of sanitization functions."""
    
    @staticmethod
    def strip_whitespace(value: str) -> str:
        """Remove leading/trailing whitespace."""
        return value.strip() if isinstance(value, str) else value
    
    @staticmethod
    def uppercase(value: str) -> str:
        """Convert to uppercase."""
        return value.upper() if isinstance(value, str) else value
    
    @staticmethod
    def lowercase(value: str) -> str:
        """Convert to lowercase."""
        return value.lower() if isinstance(value, str) else value
    
    @staticmethod
    def escape_html(value: str) -> str:
        """Escape HTML special characters."""
        return html.escape(value) if isinstance(value, str) else value
    
    @staticmethod
    def remove_null_bytes(value: str) -> str:
        """Remove null bytes (prevent null byte injection)."""
        return value.replace('\x00', '') if isinstance(value, str) else value
    
    @staticmethod
    def normalize_newlines(value: str) -> str:
        """Normalize newline characters."""
        if isinstance(value, str):
            return value.replace('\r\n', '\n').replace('\r', '\n')
        return value
    
    @staticmethod
    def remove_control_chars(value: str) -> str:
        """Remove control characters except newlines and tabs."""
        if isinstance(value, str):
            return ''.join(c for c in value if c >= ' ' or c in '\n\t')
        return value
    
    @staticmethod
    def sanitize_symbol(value: str) -> str:
        """Sanitize stock symbol."""
        if isinstance(value, str):
            # Strip, uppercase, remove non-alphanumeric except dots
            sanitized = value.strip().upper()
            sanitized = re.sub(r'[^A-Z0-9.]', '', sanitized)
            return sanitized
        return value
    
    @staticmethod
    def sanitize_for_log(value: str, max_length: int = 100) -> str:
        """Sanitize value for safe logging."""
        if isinstance(value, str):
            # Remove newlines, truncate
            sanitized = value.replace('\n', '\\n').replace('\r', '\\r')
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length] + '...'
            return sanitized
        return str(value)[:max_length]


# =============================================================================
# Strict Validator
# =============================================================================

class StrictValidator:
    """
    Strict schema-based input validator.
    
    Features:
    - Type enforcement
    - Length limits
    - Pattern matching
    - Unexpected field rejection
    - Deep nested validation
    - Sanitization pipeline
    """
    
    # Patterns for common validations
    PATTERNS = {
        'symbol': r'^[A-Z]{1,5}(\.[A-Z])?$',
        'date': r'^\d{4}-\d{2}-\d{2}$',
        'datetime': r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'alphanumeric': r'^[a-zA-Z0-9]+$',
        'alphanumeric_underscore': r'^[a-zA-Z0-9_]+$',
    }
    
    def __init__(self, schema: RequestSchema):
        self.schema = schema
    
    def validate(self, data: Any, raw_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate input data against schema.
        
        Args:
            data: Input data (dict or JSON string)
            raw_size: Size of raw input in bytes (for size limit check)
        
        Returns:
            Validated and sanitized data
        
        Raises:
            StrictValidationError: If validation fails
        """
        errors: List[Dict[str, Any]] = []
        
        # Check raw size if provided
        if raw_size is not None and raw_size > self.schema.max_size_bytes:
            errors.append({
                "field": "_request",
                "code": "SIZE_EXCEEDED",
                "message": f"Request size {raw_size} exceeds limit {self.schema.max_size_bytes}",
            })
            raise StrictValidationError(errors)
        
        # Parse JSON if string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                errors.append({
                    "field": "_request",
                    "code": "INVALID_JSON",
                    "message": f"Invalid JSON: {str(e)[:100]}",
                })
                raise StrictValidationError(errors)
        
        # Must be a dict
        if not isinstance(data, dict):
            errors.append({
                "field": "_request",
                "code": "INVALID_TYPE",
                "message": f"Expected object, got {type(data).__name__}",
            })
            raise StrictValidationError(errors)
        
        # Check field count
        if len(data) > self.schema.max_fields:
            errors.append({
                "field": "_request",
                "code": "TOO_MANY_FIELDS",
                "message": f"Too many fields: {len(data)} > {self.schema.max_fields}",
            })
            raise StrictValidationError(errors)
        
        # Check for unexpected fields
        if not self.schema.allow_extra_fields:
            extra_fields = set(data.keys()) - set(self.schema.fields.keys())
            if extra_fields:
                for field_name in extra_fields:
                    errors.append({
                        "field": field_name,
                        "code": "UNEXPECTED_FIELD",
                        "message": f"Unexpected field: '{Sanitizers.sanitize_for_log(field_name)}'",
                    })
        
        # Validate each field
        validated = {}
        for field_name, field_schema in self.schema.fields.items():
            field_errors = []
            
            if field_name in data:
                value = data[field_name]
                validated_value, field_errors = self._validate_field(
                    field_name, value, field_schema
                )
                if not field_errors:
                    validated[field_name] = validated_value
            elif field_schema.required:
                field_errors.append({
                    "field": field_name,
                    "code": "REQUIRED",
                    "message": f"Field '{field_name}' is required",
                })
            elif field_schema.default is not None:
                validated[field_name] = field_schema.default
            
            errors.extend(field_errors)
        
        if errors:
            raise StrictValidationError(errors)
        
        return validated
    
    def _validate_field(
        self, 
        field_name: str, 
        value: Any, 
        schema: FieldSchema
    ) -> tuple[Any, List[Dict[str, Any]]]:
        """Validate a single field."""
        errors = []
        
        # Check for None
        if value is None:
            if schema.required:
                errors.append({
                    "field": field_name,
                    "code": "NULL_VALUE",
                    "message": f"Field '{field_name}' cannot be null",
                })
            return None, errors
        
        # Apply sanitizer first
        if schema.sanitizer:
            try:
                value = schema.sanitizer(value)
            except Exception:
                errors.append({
                    "field": field_name,
                    "code": "SANITIZATION_FAILED",
                    "message": f"Failed to sanitize field '{field_name}'",
                })
                return value, errors
        
        # Type validation
        type_valid, value, type_error = self._validate_type(field_name, value, schema)
        if not type_valid:
            errors.append(type_error)
            return value, errors
        
        # Length validation (for strings)
        if schema.field_type == FieldType.STRING and isinstance(value, str):
            if schema.min_length is not None and len(value) < schema.min_length:
                errors.append({
                    "field": field_name,
                    "code": "TOO_SHORT",
                    "message": f"Field '{field_name}' must be at least {schema.min_length} characters",
                })
            if schema.max_length is not None and len(value) > schema.max_length:
                errors.append({
                    "field": field_name,
                    "code": "TOO_LONG",
                    "message": f"Field '{field_name}' must be at most {schema.max_length} characters",
                })
        
        # Value range validation (for numbers)
        if schema.field_type in (FieldType.INTEGER, FieldType.FLOAT):
            if schema.min_value is not None and value < schema.min_value:
                errors.append({
                    "field": field_name,
                    "code": "VALUE_TOO_LOW",
                    "message": f"Field '{field_name}' must be >= {schema.min_value}",
                })
            if schema.max_value is not None and value > schema.max_value:
                errors.append({
                    "field": field_name,
                    "code": "VALUE_TOO_HIGH",
                    "message": f"Field '{field_name}' must be <= {schema.max_value}",
                })
        
        # Pattern validation
        if schema.pattern and isinstance(value, str):
            if not re.match(schema.pattern, value):
                errors.append({
                    "field": field_name,
                    "code": "PATTERN_MISMATCH",
                    "message": f"Field '{field_name}' does not match required pattern",
                })
        
        # Choices validation
        if schema.choices is not None and value not in schema.choices:
            safe_choices = [str(c)[:20] for c in schema.choices[:10]]
            errors.append({
                "field": field_name,
                "code": "INVALID_CHOICE",
                "message": f"Field '{field_name}' must be one of: {safe_choices}",
            })
        
        # Custom validator
        if schema.validator and not errors:
            try:
                if not schema.validator(value):
                    errors.append({
                        "field": field_name,
                        "code": "CUSTOM_VALIDATION_FAILED",
                        "message": f"Field '{field_name}' failed validation",
                    })
            except Exception:
                errors.append({
                    "field": field_name,
                    "code": "VALIDATION_ERROR",
                    "message": f"Error validating field '{field_name}'",
                })
        
        # List validation
        if schema.field_type == FieldType.LIST and isinstance(value, list):
            if schema.max_length is not None and len(value) > schema.max_length:
                errors.append({
                    "field": field_name,
                    "code": "LIST_TOO_LONG",
                    "message": f"List '{field_name}' exceeds max length {schema.max_length}",
                })
            
            if schema.item_schema and not errors:
                validated_items = []
                for i, item in enumerate(value):
                    item_value, item_errors = self._validate_field(
                        f"{field_name}[{i}]", item, schema.item_schema
                    )
                    errors.extend(item_errors)
                    if not item_errors:
                        validated_items.append(item_value)
                value = validated_items
        
        # Nested dict validation
        if schema.field_type == FieldType.DICT and schema.nested_schema:
            nested_validator = StrictValidator(schema.nested_schema)
            try:
                value = nested_validator.validate(value)
            except StrictValidationError as e:
                for error in e.errors:
                    error["field"] = f"{field_name}.{error['field']}"
                    errors.append(error)
        
        return value, errors
    
    def _validate_type(
        self, 
        field_name: str, 
        value: Any, 
        schema: FieldSchema
    ) -> tuple[bool, Any, Optional[Dict]]:
        """Validate and coerce field type."""
        field_type = schema.field_type
        
        try:
            if field_type == FieldType.STRING:
                if not isinstance(value, str):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected string, got {type(value).__name__}",
                    }
                # Apply default string sanitization
                value = Sanitizers.remove_null_bytes(value)
                value = Sanitizers.remove_control_chars(value)
                
            elif field_type == FieldType.INTEGER:
                if isinstance(value, bool):  # bool is subclass of int
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected integer, got boolean",
                    }
                if not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        return False, value, {
                            "field": field_name,
                            "code": "INVALID_TYPE",
                            "message": f"Expected integer, got {type(value).__name__}",
                        }
                        
            elif field_type == FieldType.FLOAT:
                if isinstance(value, bool):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected float, got boolean",
                    }
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        return False, value, {
                            "field": field_name,
                            "code": "INVALID_TYPE",
                            "message": f"Expected float, got {type(value).__name__}",
                        }
                else:
                    value = float(value)
                    
            elif field_type == FieldType.BOOLEAN:
                if not isinstance(value, bool):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected boolean, got {type(value).__name__}",
                    }
                    
            elif field_type == FieldType.DATE:
                if isinstance(value, str):
                    if not re.match(self.PATTERNS['date'], value):
                        return False, value, {
                            "field": field_name,
                            "code": "INVALID_FORMAT",
                            "message": f"Date must be YYYY-MM-DD format",
                        }
                    try:
                        datetime.strptime(value, '%Y-%m-%d')
                    except ValueError:
                        return False, value, {
                            "field": field_name,
                            "code": "INVALID_DATE",
                            "message": f"Invalid date value",
                        }
                elif isinstance(value, date):
                    value = value.isoformat()
                else:
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected date string, got {type(value).__name__}",
                    }
                    
            elif field_type == FieldType.SYMBOL:
                if not isinstance(value, str):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected string, got {type(value).__name__}",
                    }
                value = Sanitizers.sanitize_symbol(value)
                if not re.match(self.PATTERNS['symbol'], value):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_SYMBOL",
                        "message": f"Invalid stock symbol format",
                    }
                    
            elif field_type == FieldType.LIST:
                if not isinstance(value, list):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected list, got {type(value).__name__}",
                    }
                    
            elif field_type == FieldType.DICT:
                if not isinstance(value, dict):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected object, got {type(value).__name__}",
                    }
                    
            elif field_type == FieldType.EMAIL:
                if not isinstance(value, str):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_TYPE",
                        "message": f"Expected string, got {type(value).__name__}",
                    }
                value = value.lower().strip()
                if not re.match(self.PATTERNS['email'], value):
                    return False, value, {
                        "field": field_name,
                        "code": "INVALID_EMAIL",
                        "message": f"Invalid email format",
                    }
            
            return True, value, None
            
        except Exception as e:
            return False, value, {
                "field": field_name,
                "code": "TYPE_CONVERSION_ERROR",
                "message": f"Failed to validate type for field '{field_name}'",
            }


# =============================================================================
# Pre-defined Schemas for Stock Analyzer API
# =============================================================================

# Analyze Request Schema
AnalyzeRequestSchema = RequestSchema(
    name="AnalyzeRequest",
    fields={
        "symbol": FieldSchema(
            name="symbol",
            field_type=FieldType.SYMBOL,
            required=True,
            min_length=1,
            max_length=10,
            sanitizer=Sanitizers.sanitize_symbol,
            description="Stock ticker symbol (e.g., AAPL)",
        ),
        "start": FieldSchema(
            name="start",
            field_type=FieldType.DATE,
            required=False,
            description="Start date YYYY-MM-DD",
        ),
        "end": FieldSchema(
            name="end",
            field_type=FieldType.DATE,
            required=False,
            description="End date YYYY-MM-DD",
        ),
        "horizon": FieldSchema(
            name="horizon",
            field_type=FieldType.INTEGER,
            required=False,
            default=5,
            min_value=1,
            max_value=60,
            description="Prediction horizon in days",
        ),
        "threshold": FieldSchema(
            name="threshold",
            field_type=FieldType.FLOAT,
            required=False,
            default=0.55,
            min_value=0.5,
            max_value=0.95,
            description="Signal threshold (0.5-0.95)",
        ),
        "timeframe": FieldSchema(
            name="timeframe",
            field_type=FieldType.STRING,
            required=False,
            default="1Day",
            choices=["1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"],
            description="Bar timeframe",
        ),
    },
    allow_extra_fields=False,
    max_fields=10,
    max_size_bytes=10_000,  # 10KB
)


# Batch Analyze Request Schema
BatchAnalyzeRequestSchema = RequestSchema(
    name="BatchAnalyzeRequest",
    fields={
        "symbols": FieldSchema(
            name="symbols",
            field_type=FieldType.LIST,
            required=True,
            min_length=1,
            max_length=100,
            item_schema=FieldSchema(
                name="symbol",
                field_type=FieldType.SYMBOL,
                sanitizer=Sanitizers.sanitize_symbol,
            ),
            description="List of stock symbols",
        ),
        "start": FieldSchema(
            name="start",
            field_type=FieldType.DATE,
            required=False,
            description="Start date YYYY-MM-DD",
        ),
        "end": FieldSchema(
            name="end",
            field_type=FieldType.DATE,
            required=False,
            description="End date YYYY-MM-DD",
        ),
        "horizon": FieldSchema(
            name="horizon",
            field_type=FieldType.INTEGER,
            required=False,
            default=5,
            min_value=1,
            max_value=60,
            description="Prediction horizon in days",
        ),
        "threshold": FieldSchema(
            name="threshold",
            field_type=FieldType.FLOAT,
            required=False,
            default=0.55,
            min_value=0.5,
            max_value=0.95,
            description="Signal threshold",
        ),
    },
    allow_extra_fields=False,
    max_fields=10,
    max_size_bytes=50_000,  # 50KB
)


# Screener Filters Schema
ScreenerFiltersSchema = RequestSchema(
    name="ScreenerFilters",
    fields={
        "rsi_below": FieldSchema(
            name="rsi_below",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0,
            max_value=100,
        ),
        "rsi_above": FieldSchema(
            name="rsi_above",
            field_type=FieldType.FLOAT,
            required=False,
            min_value=0,
            max_value=100,
        ),
        "macd_bullish": FieldSchema(
            name="macd_bullish",
            field_type=FieldType.BOOLEAN,
            required=False,
        ),
        "macd_bearish": FieldSchema(
            name="macd_bearish",
            field_type=FieldType.BOOLEAN,
            required=False,
        ),
        "above_sma_20": FieldSchema(
            name="above_sma_20",
            field_type=FieldType.BOOLEAN,
            required=False,
        ),
        "above_sma_50": FieldSchema(
            name="above_sma_50",
            field_type=FieldType.BOOLEAN,
            required=False,
        ),
        "above_sma_200": FieldSchema(
            name="above_sma_200",
            field_type=FieldType.BOOLEAN,
            required=False,
        ),
        "volume_above_avg": FieldSchema(
            name="volume_above_avg",
            field_type=FieldType.BOOLEAN,
            required=False,
        ),
        "near_52w_high": FieldSchema(
            name="near_52w_high",
            field_type=FieldType.BOOLEAN,
            required=False,
        ),
        "near_52w_low": FieldSchema(
            name="near_52w_low",
            field_type=FieldType.BOOLEAN,
            required=False,
        ),
    },
    allow_extra_fields=False,
    max_fields=20,
)


# Screener Request Schema
ScreenerRequestSchema = RequestSchema(
    name="ScreenerRequest",
    fields={
        "symbols": FieldSchema(
            name="symbols",
            field_type=FieldType.LIST,
            required=False,
            max_length=500,
            item_schema=FieldSchema(
                name="symbol",
                field_type=FieldType.SYMBOL,
                sanitizer=Sanitizers.sanitize_symbol,
            ),
            description="List of symbols to scan (or empty for all)",
        ),
        "filters": FieldSchema(
            name="filters",
            field_type=FieldType.DICT,
            required=False,
            nested_schema=ScreenerFiltersSchema,
            description="Filter criteria",
        ),
        "limit": FieldSchema(
            name="limit",
            field_type=FieldType.INTEGER,
            required=False,
            default=50,
            min_value=1,
            max_value=500,
            description="Max results to return",
        ),
        "lookback": FieldSchema(
            name="lookback",
            field_type=FieldType.INTEGER,
            required=False,
            default=365,
            min_value=30,
            max_value=1000,
            description="Lookback period in days",
        ),
    },
    allow_extra_fields=False,
    max_fields=10,
    max_size_bytes=100_000,  # 100KB
)


# =============================================================================
# Helper Functions
# =============================================================================

def validate_analyze_request(data: Any, raw_size: Optional[int] = None) -> Dict[str, Any]:
    """Validate analyze request with strict schema."""
    validator = StrictValidator(AnalyzeRequestSchema)
    return validator.validate(data, raw_size)


def validate_batch_analyze_request(data: Any, raw_size: Optional[int] = None) -> Dict[str, Any]:
    """Validate batch analyze request with strict schema."""
    validator = StrictValidator(BatchAnalyzeRequestSchema)
    return validator.validate(data, raw_size)


def validate_screener_request(data: Any, raw_size: Optional[int] = None) -> Dict[str, Any]:
    """Validate screener request with strict schema."""
    validator = StrictValidator(ScreenerRequestSchema)
    return validator.validate(data, raw_size)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Exceptions
    "StrictValidationError",
    # Core classes
    "FieldType",
    "FieldSchema",
    "RequestSchema",
    "StrictValidator",
    "Sanitizers",
    # Pre-defined schemas
    "AnalyzeRequestSchema",
    "BatchAnalyzeRequestSchema",
    "ScreenerRequestSchema",
    "ScreenerFiltersSchema",
    # Helper functions
    "validate_analyze_request",
    "validate_batch_analyze_request",
    "validate_screener_request",
]
