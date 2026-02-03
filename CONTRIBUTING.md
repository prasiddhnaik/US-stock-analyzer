# Pre-Push Protocol

Follow this checklist before pushing code to GitHub to ensure quality and prevent deployment failures.

## Quick Checklist

```bash
# Run this one-liner before every push:
source venv/bin/activate && \
python -m py_compile app.py charts.py indicators.py && \
pytest tests/ -q && \
echo "✅ Ready to push"
```

---

## Full Protocol

### 1. Verify Virtual Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Verify you're in the right environment
which python
# Should show: /Users/.../stock ana/venv/bin/python
```

### 2. Syntax Check (All Modified Files)

```bash
# Check Python syntax compiles
python -m py_compile app.py
python -m py_compile charts.py
python -m py_compile indicators.py
python -m py_compile api_server.py
python -m py_compile security.py

# Or check all Python files at once:
find . -name "*.py" -not -path "./venv/*" -not -path "./frontend/*" | xargs python -m py_compile
```

### 3. Import Verification

```bash
# Verify all imports work (catches missing dependencies)
python -c "from app import main; print('✅ app.py imports OK')"
python -c "from charts import create_price_chart, create_rsi_chart, create_stochastic_chart; print('✅ charts.py imports OK')"
python -c "from indicators import compute_all_indicators; print('✅ indicators.py imports OK')"
python -c "from api_server import app; print('✅ api_server.py imports OK')"
```

### 4. Run Unit Tests

```bash
# Run all tests (quick mode)
pytest tests/ -q

# Run with verbose output (if debugging)
pytest tests/ -v

# Run specific test file
pytest tests/test_indicators.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

**Expected output:** All tests should pass (currently 207+ tests)

### 5. Check Dependencies

```bash
# Verify all imports in requirements.txt are used
pip freeze | grep -E "scipy|pandas|numpy|plotly|streamlit|ta|scikit-learn|alpaca"

# Ensure requirements.txt has all needed packages
cat requirements.txt
```

**Key dependencies to verify:**
- `scipy` - Used by indicators.py
- `ta` - Technical analysis library
- `alpaca-py` - Market data API
- `plotly` - Charts
- `streamlit` - UI

### 6. Linting (Optional but Recommended)

```bash
# Run Black formatter (check only, don't modify)
black --check .

# Run flake8 linter
flake8 . --max-line-length=120 --exclude=venv,frontend

# Run isort (import sorter)
isort --check-only .
```

### 7. Security Check

```bash
# Run bandit security scanner
bandit -r . -x ./venv,./frontend,./tests -ll

# Check for known vulnerabilities in dependencies
pip-audit 2>/dev/null || echo "pip-audit not installed (optional)"
```

### 8. Git Status Review

```bash
# Check what will be committed
git status

# Review changes
git diff --stat

# Ensure no secrets are staged
git diff --cached | grep -i "api_key\|secret\|password" && echo "⚠️ SECRETS DETECTED" || echo "✅ No secrets"
```

### 9. Commit and Push

```bash
# Stage changes
git add -A

# Commit with descriptive message
git commit -m "Brief description of changes"

# Push to remote
git push origin main
```

---

## CI/CD Verification

After pushing, verify GitHub Actions runs successfully:

1. Go to: `https://github.com/prasiddhnaik/US-stock-analyzer/actions`
2. Check the latest workflow run
3. All jobs should pass:
   - ✅ Lint (black, isort, flake8)
   - ✅ Test (pytest on Python 3.10, 3.11, 3.12)
   - ✅ Security (bandit)
   - ✅ Build (syntax verification)

---

## Streamlit Cloud Deployment

After GitHub Actions passes:

1. Streamlit Cloud auto-deploys from `main` branch
2. If issues occur:
   - Go to app dashboard
   - Click "Manage app" → "Reboot app"
   - Check logs for errors

---

## Common Issues & Fixes

### ImportError on Streamlit Cloud

**Cause:** Missing dependency in `requirements.txt`

**Fix:**
```bash
# Find what's imported
grep -r "^from \|^import " *.py | grep -v venv

# Add missing package to requirements.txt
echo "package-name>=version" >> requirements.txt
```

### Tests Fail Locally but Pass in CI

**Cause:** Different Python version or environment

**Fix:**
```bash
# Check Python version matches CI (3.10-3.12)
python --version

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Syntax Error After Push

**Cause:** Didn't run `py_compile` before push

**Fix:** Always run the quick checklist before pushing.

---

## Automated Pre-Push Hook (Optional)

Add this to `.git/hooks/pre-push`:

```bash
#!/bin/bash
echo "🔍 Running pre-push checks..."

# Activate venv
source venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null

# Syntax check
echo "Checking syntax..."
find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./frontend/*" | xargs python -m py_compile
if [ $? -ne 0 ]; then
    echo "❌ Syntax error detected. Push aborted."
    exit 1
fi

# Run tests
echo "Running tests..."
pytest tests/ -q
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Push aborted."
    exit 1
fi

echo "✅ All checks passed. Proceeding with push."
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-push
```

---

## Summary

| Step | Command | Must Pass |
|------|---------|-----------|
| Syntax | `python -m py_compile *.py` | ✅ Yes |
| Imports | `python -c "from app import main"` | ✅ Yes |
| Tests | `pytest tests/ -q` | ✅ Yes |
| Lint | `black --check .` | ⚠️ Recommended |
| Security | `bandit -r . -ll` | ⚠️ Recommended |
| Secrets | `git diff --cached \| grep secret` | ✅ Yes |
