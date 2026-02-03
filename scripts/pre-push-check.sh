#!/bin/bash
# Pre-Push Quality Check Script
# Run this before pushing to GitHub

set -e  # Exit on first error

echo "════════════════════════════════════════════════════════════════"
echo "  📋 PRE-PUSH QUALITY CHECK"
echo "════════════════════════════════════════════════════════════════"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/.."

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo -e "${RED}❌ No virtual environment found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Step 1: Syntax Check
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  1️⃣  SYNTAX CHECK"
echo "════════════════════════════════════════════════════════════════"

SYNTAX_ERROR=0
for file in app.py charts.py indicators.py api_server.py security.py data_fetcher.py model.py; do
    if [ -f "$file" ]; then
        if python -m py_compile "$file" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $file"
        else
            echo -e "${RED}✗${NC} $file"
            SYNTAX_ERROR=1
        fi
    fi
done

if [ $SYNTAX_ERROR -eq 1 ]; then
    echo -e "${RED}❌ Syntax errors detected. Fix before pushing.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All syntax checks passed${NC}"

# Step 2: Import Verification
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  2️⃣  IMPORT VERIFICATION"
echo "════════════════════════════════════════════════════════════════"

IMPORT_ERROR=0

python -c "from app import main" 2>/dev/null && echo -e "${GREEN}✓${NC} app.py imports" || { echo -e "${RED}✗${NC} app.py imports"; IMPORT_ERROR=1; }
python -c "from charts import create_price_chart, create_rsi_chart, create_stochastic_chart" 2>/dev/null && echo -e "${GREEN}✓${NC} charts.py imports" || { echo -e "${RED}✗${NC} charts.py imports"; IMPORT_ERROR=1; }
python -c "from indicators import compute_all_indicators" 2>/dev/null && echo -e "${GREEN}✓${NC} indicators.py imports" || { echo -e "${RED}✗${NC} indicators.py imports"; IMPORT_ERROR=1; }
python -c "from api_server import app" 2>/dev/null && echo -e "${GREEN}✓${NC} api_server.py imports" || { echo -e "${RED}✗${NC} api_server.py imports"; IMPORT_ERROR=1; }

if [ $IMPORT_ERROR -eq 1 ]; then
    echo -e "${RED}❌ Import errors detected. Check requirements.txt${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All imports verified${NC}"

# Step 3: Unit Tests
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  3️⃣  UNIT TESTS"
echo "════════════════════════════════════════════════════════════════"

if pytest tests/ -q --tb=no; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}❌ Tests failed. Fix before pushing.${NC}"
    exit 1
fi

# Step 4: Secret Detection
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  4️⃣  SECRET DETECTION"
echo "════════════════════════════════════════════════════════════════"

if git diff --cached 2>/dev/null | grep -iE "(api_key|api_secret|password|token).*=" | grep -v "\.example\|\.template\|getenv\|os\.environ" > /dev/null; then
    echo -e "${RED}⚠️  WARNING: Potential secrets detected in staged files${NC}"
    echo "Review your changes carefully before pushing."
else
    echo -e "${GREEN}✓ No obvious secrets detected${NC}"
fi

# Step 5: Git Status
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  5️⃣  GIT STATUS"
echo "════════════════════════════════════════════════════════════════"

echo "Branch: $(git branch --show-current)"
echo "Changes to be committed:"
git diff --stat HEAD 2>/dev/null | tail -5

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}✅ ALL CHECKS PASSED - READY TO PUSH${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Run these commands to push:"
echo "  git add -A"
echo "  git commit -m \"Your commit message\""
echo "  git push origin main"
echo ""
