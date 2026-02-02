"""
Unit tests for strict_validator.py
Tests schema-based validation, type checks, and sanitization.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strict_validator import (
    StrictValidationError,
    StrictValidator,
    FieldType,
    FieldSchema,
    RequestSchema,
    Sanitizers,
    AnalyzeRequestSchema,
    BatchAnalyzeRequestSchema,
    ScreenerRequestSchema,
    validate_analyze_request,
    validate_screener_request,
)


# =============================================================================
# Sanitizer Tests
# =============================================================================

class TestSanitizers:
    """Tests for sanitization functions."""
    
    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        assert Sanitizers.strip_whitespace("  hello  ") == "hello"
        assert Sanitizers.strip_whitespace("\t\ntest\n\t") == "test"
    
    def test_uppercase(self):
        """Test uppercase conversion."""
        assert Sanitizers.uppercase("aapl") == "AAPL"
        assert Sanitizers.uppercase("MiXeD") == "MIXED"
    
    def test_sanitize_symbol(self):
        """Test symbol sanitization."""
        assert Sanitizers.sanitize_symbol("aapl") == "AAPL"
        assert Sanitizers.sanitize_symbol("  msft  ") == "MSFT"
        assert Sanitizers.sanitize_symbol("brk.a") == "BRK.A"
        assert Sanitizers.sanitize_symbol("test@#$") == "TEST"
    
    def test_remove_null_bytes(self):
        """Test null byte removal."""
        assert Sanitizers.remove_null_bytes("hello\x00world") == "helloworld"
    
    def test_escape_html(self):
        """Test HTML escaping."""
        assert Sanitizers.escape_html("<script>") == "&lt;script&gt;"
        assert Sanitizers.escape_html("a & b") == "a &amp; b"
    
    def test_remove_control_chars(self):
        """Test control character removal."""
        # Keep newlines and tabs, remove others
        result = Sanitizers.remove_control_chars("hello\x00\x01\nworld\t!")
        assert result == "hello\nworld\t!"


# =============================================================================
# Type Validation Tests
# =============================================================================

class TestTypeValidation:
    """Tests for type validation."""
    
    def test_string_type(self):
        """Test string type validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "name": FieldSchema(name="name", field_type=FieldType.STRING, required=True)
            }
        )
        validator = StrictValidator(schema)
        
        # Valid string
        result = validator.validate({"name": "test"})
        assert result["name"] == "test"
        
        # Invalid: number instead of string
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"name": 123})
        assert any(e["code"] == "INVALID_TYPE" for e in exc.value.errors)
    
    def test_integer_type(self):
        """Test integer type validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "count": FieldSchema(name="count", field_type=FieldType.INTEGER, required=True)
            }
        )
        validator = StrictValidator(schema)
        
        # Valid integer
        result = validator.validate({"count": 42})
        assert result["count"] == 42
        
        # Boolean should NOT be accepted as integer
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"count": True})
        assert any(e["code"] == "INVALID_TYPE" for e in exc.value.errors)
    
    def test_float_type(self):
        """Test float type validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "price": FieldSchema(name="price", field_type=FieldType.FLOAT, required=True)
            }
        )
        validator = StrictValidator(schema)
        
        # Valid float
        result = validator.validate({"price": 123.45})
        assert result["price"] == 123.45
        
        # Integer should be converted to float
        result = validator.validate({"price": 100})
        assert result["price"] == 100.0
    
    def test_boolean_type(self):
        """Test boolean type validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "active": FieldSchema(name="active", field_type=FieldType.BOOLEAN, required=True)
            }
        )
        validator = StrictValidator(schema)
        
        # Valid boolean
        result = validator.validate({"active": True})
        assert result["active"] is True
        
        # String "true" should NOT be accepted
        with pytest.raises(StrictValidationError):
            validator.validate({"active": "true"})
    
    def test_date_type(self):
        """Test date type validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "date": FieldSchema(name="date", field_type=FieldType.DATE, required=True)
            }
        )
        validator = StrictValidator(schema)
        
        # Valid date
        result = validator.validate({"date": "2024-01-15"})
        assert result["date"] == "2024-01-15"
        
        # Invalid format
        with pytest.raises(StrictValidationError):
            validator.validate({"date": "01/15/2024"})
        
        # Invalid date
        with pytest.raises(StrictValidationError):
            validator.validate({"date": "2024-13-45"})
    
    def test_symbol_type(self):
        """Test symbol type validation and sanitization."""
        schema = RequestSchema(
            name="Test",
            fields={
                "symbol": FieldSchema(name="symbol", field_type=FieldType.SYMBOL, required=True)
            }
        )
        validator = StrictValidator(schema)
        
        # Valid symbol (gets uppercased)
        result = validator.validate({"symbol": "aapl"})
        assert result["symbol"] == "AAPL"
        
        # Invalid symbol
        with pytest.raises(StrictValidationError):
            validator.validate({"symbol": "INVALID123456"})


# =============================================================================
# Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """Tests for schema-based validation."""
    
    def test_required_field_missing(self):
        """Test required field validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "required_field": FieldSchema(
                    name="required_field", 
                    field_type=FieldType.STRING, 
                    required=True
                )
            }
        )
        validator = StrictValidator(schema)
        
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({})
        assert any(e["code"] == "REQUIRED" for e in exc.value.errors)
    
    def test_unexpected_field_rejected(self):
        """Test unexpected fields are rejected."""
        schema = RequestSchema(
            name="Test",
            fields={
                "allowed": FieldSchema(name="allowed", field_type=FieldType.STRING)
            },
            allow_extra_fields=False
        )
        validator = StrictValidator(schema)
        
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"allowed": "ok", "unexpected": "bad"})
        assert any(e["code"] == "UNEXPECTED_FIELD" for e in exc.value.errors)
    
    def test_length_limits(self):
        """Test string length limits."""
        schema = RequestSchema(
            name="Test",
            fields={
                "name": FieldSchema(
                    name="name", 
                    field_type=FieldType.STRING,
                    min_length=3,
                    max_length=10
                )
            }
        )
        validator = StrictValidator(schema)
        
        # Too short
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"name": "ab"})
        assert any(e["code"] == "TOO_SHORT" for e in exc.value.errors)
        
        # Too long
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"name": "verylongname"})
        assert any(e["code"] == "TOO_LONG" for e in exc.value.errors)
        
        # Just right
        result = validator.validate({"name": "valid"})
        assert result["name"] == "valid"
    
    def test_value_range(self):
        """Test numeric value range limits."""
        schema = RequestSchema(
            name="Test",
            fields={
                "count": FieldSchema(
                    name="count",
                    field_type=FieldType.INTEGER,
                    min_value=1,
                    max_value=100
                )
            }
        )
        validator = StrictValidator(schema)
        
        # Too low
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"count": 0})
        assert any(e["code"] == "VALUE_TOO_LOW" for e in exc.value.errors)
        
        # Too high
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"count": 101})
        assert any(e["code"] == "VALUE_TOO_HIGH" for e in exc.value.errors)
    
    def test_choices_validation(self):
        """Test enum/choices validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "status": FieldSchema(
                    name="status",
                    field_type=FieldType.STRING,
                    choices=["active", "inactive", "pending"]
                )
            }
        )
        validator = StrictValidator(schema)
        
        # Valid choice
        result = validator.validate({"status": "active"})
        assert result["status"] == "active"
        
        # Invalid choice
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"status": "invalid"})
        assert any(e["code"] == "INVALID_CHOICE" for e in exc.value.errors)
    
    def test_pattern_validation(self):
        """Test regex pattern validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "code": FieldSchema(
                    name="code",
                    field_type=FieldType.STRING,
                    pattern=r"^[A-Z]{3}-\d{3}$"
                )
            }
        )
        validator = StrictValidator(schema)
        
        # Valid pattern
        result = validator.validate({"code": "ABC-123"})
        assert result["code"] == "ABC-123"
        
        # Invalid pattern
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"code": "invalid"})
        assert any(e["code"] == "PATTERN_MISMATCH" for e in exc.value.errors)
    
    def test_default_values(self):
        """Test default value assignment."""
        schema = RequestSchema(
            name="Test",
            fields={
                "optional": FieldSchema(
                    name="optional",
                    field_type=FieldType.INTEGER,
                    required=False,
                    default=42
                )
            }
        )
        validator = StrictValidator(schema)
        
        result = validator.validate({})
        assert result["optional"] == 42
    
    def test_max_fields_limit(self):
        """Test maximum fields limit."""
        schema = RequestSchema(
            name="Test",
            fields={f"field{i}": FieldSchema(name=f"field{i}", field_type=FieldType.STRING) for i in range(5)},
            max_fields=3
        )
        validator = StrictValidator(schema)
        
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"field0": "a", "field1": "b", "field2": "c", "field3": "d"})
        assert any(e["code"] == "TOO_MANY_FIELDS" for e in exc.value.errors)


# =============================================================================
# List Validation Tests
# =============================================================================

class TestListValidation:
    """Tests for list validation."""
    
    def test_list_type(self):
        """Test list type validation."""
        schema = RequestSchema(
            name="Test",
            fields={
                "items": FieldSchema(
                    name="items",
                    field_type=FieldType.LIST,
                    item_schema=FieldSchema(name="item", field_type=FieldType.STRING)
                )
            }
        )
        validator = StrictValidator(schema)
        
        result = validator.validate({"items": ["a", "b", "c"]})
        assert result["items"] == ["a", "b", "c"]
    
    def test_list_length_limit(self):
        """Test list length limit."""
        schema = RequestSchema(
            name="Test",
            fields={
                "items": FieldSchema(
                    name="items",
                    field_type=FieldType.LIST,
                    max_length=3
                )
            }
        )
        validator = StrictValidator(schema)
        
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"items": [1, 2, 3, 4, 5]})
        assert any(e["code"] == "LIST_TOO_LONG" for e in exc.value.errors)
    
    def test_list_item_validation(self):
        """Test validation of list items."""
        schema = RequestSchema(
            name="Test",
            fields={
                "symbols": FieldSchema(
                    name="symbols",
                    field_type=FieldType.LIST,
                    item_schema=FieldSchema(
                        name="symbol",
                        field_type=FieldType.SYMBOL,
                        sanitizer=Sanitizers.sanitize_symbol
                    )
                )
            }
        )
        validator = StrictValidator(schema)
        
        # Valid symbols
        result = validator.validate({"symbols": ["aapl", "msft", "goog"]})
        assert result["symbols"] == ["AAPL", "MSFT", "GOOG"]


# =============================================================================
# Predefined Schema Tests
# =============================================================================

class TestAnalyzeRequestSchema:
    """Tests for AnalyzeRequest schema."""
    
    def test_valid_request(self):
        """Test valid analyze request."""
        result = validate_analyze_request({
            "symbol": "AAPL",
            "horizon": 5,
            "threshold": 0.6
        })
        assert result["symbol"] == "AAPL"
        assert result["horizon"] == 5
        assert result["threshold"] == 0.6
    
    def test_symbol_sanitization(self):
        """Test symbol is sanitized."""
        result = validate_analyze_request({
            "symbol": "  aapl  "
        })
        assert result["symbol"] == "AAPL"
    
    def test_invalid_symbol_rejected(self):
        """Test invalid symbol is rejected."""
        with pytest.raises(StrictValidationError):
            validate_analyze_request({"symbol": "INVALID123456789"})
    
    def test_horizon_range(self):
        """Test horizon range validation."""
        # Too low
        with pytest.raises(StrictValidationError):
            validate_analyze_request({"symbol": "AAPL", "horizon": 0})
        
        # Too high
        with pytest.raises(StrictValidationError):
            validate_analyze_request({"symbol": "AAPL", "horizon": 100})
    
    def test_unexpected_field_rejected(self):
        """Test unexpected fields are rejected."""
        with pytest.raises(StrictValidationError) as exc:
            validate_analyze_request({
                "symbol": "AAPL",
                "malicious_field": "payload"
            })
        assert any(e["code"] == "UNEXPECTED_FIELD" for e in exc.value.errors)
    
    def test_defaults_applied(self):
        """Test default values are applied."""
        result = validate_analyze_request({"symbol": "AAPL"})
        assert result["horizon"] == 5  # default
        assert result["threshold"] == 0.55  # default


class TestScreenerRequestSchema:
    """Tests for ScreenerRequest schema."""
    
    def test_valid_request(self):
        """Test valid screener request."""
        result = validate_screener_request({
            "symbols": ["AAPL", "MSFT"],
            "limit": 50
        })
        assert result["symbols"] == ["AAPL", "MSFT"]
        assert result["limit"] == 50
    
    def test_filters_nested_validation(self):
        """Test nested filters validation."""
        result = validate_screener_request({
            "filters": {
                "rsi_below": 30,
                "macd_bullish": True
            }
        })
        assert result["filters"]["rsi_below"] == 30
        assert result["filters"]["macd_bullish"] is True
    
    def test_invalid_filter_rejected(self):
        """Test invalid filter values are rejected."""
        with pytest.raises(StrictValidationError):
            validate_screener_request({
                "filters": {
                    "rsi_below": 150  # > 100
                }
            })


# =============================================================================
# Security Tests
# =============================================================================

class TestSecurityValidation:
    """Tests for security-related validation."""
    
    def test_null_byte_injection_blocked(self):
        """Test null bytes are removed from strings."""
        schema = RequestSchema(
            name="Test",
            fields={
                "name": FieldSchema(name="name", field_type=FieldType.STRING)
            }
        )
        validator = StrictValidator(schema)
        
        result = validator.validate({"name": "hello\x00world"})
        assert "\x00" not in result["name"]
        assert result["name"] == "helloworld"
    
    def test_control_char_injection_blocked(self):
        """Test control characters are removed."""
        schema = RequestSchema(
            name="Test",
            fields={
                "name": FieldSchema(name="name", field_type=FieldType.STRING)
            }
        )
        validator = StrictValidator(schema)
        
        result = validator.validate({"name": "hello\x01\x02world"})
        assert result["name"] == "helloworld"
    
    def test_request_size_limit(self):
        """Test request size limit."""
        schema = RequestSchema(
            name="Test",
            fields={
                "data": FieldSchema(name="data", field_type=FieldType.STRING)
            },
            max_size_bytes=100
        )
        validator = StrictValidator(schema)
        
        with pytest.raises(StrictValidationError) as exc:
            validator.validate({"data": "x"}, raw_size=1000)
        assert any(e["code"] == "SIZE_EXCEEDED" for e in exc.value.errors)
    
    def test_json_parsing_error(self):
        """Test invalid JSON is rejected."""
        schema = RequestSchema(
            name="Test",
            fields={}
        )
        validator = StrictValidator(schema)
        
        with pytest.raises(StrictValidationError) as exc:
            validator.validate("{invalid json}")
        assert any(e["code"] == "INVALID_JSON" for e in exc.value.errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
