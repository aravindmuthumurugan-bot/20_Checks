#!/bin/bash

################################################################################
# Fix _lzma Module Issue for Python 3.10.13
# This script resolves the "ModuleNotFoundError: No module named '_lzma'" error
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║      Fix _lzma Module Issue for Python 3.10.13                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${YELLOW}This script will:${NC}"
echo "  1. Install liblzma-dev system package"
echo "  2. Rebuild Python 3.10.13 with lzma support"
echo "  3. Reinstall all your packages"
echo ""
echo -e "${RED}WARNING: This will take 10-15 minutes${NC}"
echo -e "${YELLOW}Press ENTER to continue or Ctrl+C to cancel${NC}"
read

# ==================== CHECK PREREQUISITES ====================

echo ""
echo -e "${BLUE}Step 1: Checking Prerequisites${NC}"

# Check if pyenv is available
if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}✗ pyenv not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} pyenv is installed"

CURRENT_PYTHON=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Current Python: $CURRENT_PYTHON"

# ==================== BACKUP PACKAGES ====================

echo ""
echo -e "${BLUE}Step 2: Backing Up Package List${NC}"

BACKUP_DIR="lzma_fix_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
pip freeze > "$BACKUP_DIR/requirements.txt"
echo -e "${GREEN}✓${NC} Package list saved to: $BACKUP_DIR/requirements.txt"

# ==================== INSTALL SYSTEM DEPENDENCIES ====================

echo ""
echo -e "${BLUE}Step 3: Installing System Dependencies${NC}"

echo -e "${YELLOW}Installing liblzma-dev and other build dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    zlib1g-dev

echo -e "${GREEN}✓${NC} System dependencies installed"

# ==================== REBUILD PYTHON ====================

echo ""
echo -e "${BLUE}Step 4: Rebuilding Python 3.10.13 with lzma Support${NC}"

echo -e "${YELLOW}This will take several minutes...${NC}"

# Uninstall current Python version
pyenv uninstall -f 3.10.13

# Reinstall Python with proper flags
CFLAGS="-I/usr/include" \
LDFLAGS="-L/usr/lib/x86_64-linux-gnu" \
pyenv install 3.10.13

echo -e "${GREEN}✓${NC} Python 3.10.13 rebuilt with lzma support"

# Set the version
pyenv global 3.10.13
pyenv rehash

# ==================== VERIFY LZMA ====================

echo ""
echo -e "${BLUE}Step 5: Verifying lzma Module${NC}"

python3 << 'PYEOF'
try:
    import lzma
    print("✓ lzma module is now available")
except ImportError as e:
    print(f"✗ lzma module still not available: {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to import lzma module${NC}"
    exit 1
fi

# ==================== REINSTALL PACKAGES ====================

echo ""
echo -e "${BLUE}Step 6: Reinstalling All Packages${NC}"

echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

echo -e "${YELLOW}Installing packages from backup...${NC}"
pip install --break-system-packages -r "$BACKUP_DIR/requirements.txt"

echo -e "${GREEN}✓${NC} All packages reinstalled"

# ==================== FINAL VERIFICATION ====================

echo ""
echo -e "${BLUE}Step 7: Final Verification${NC}"

python3 << 'PYEOF'
import sys
print("\nPython version:", sys.version)

# Test critical imports
print("\nTesting imports:")

try:
    import lzma
    print("  ✓ lzma")
except ImportError as e:
    print(f"  ✗ lzma: {e}")
    sys.exit(1)

try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ torch: {e}")

try:
    import tensorflow as tf
    print(f"  ✓ tensorflow {tf.__version__}")
except ImportError as e:
    print(f"  ✗ tensorflow: {e}")

try:
    import easyocr
    print(f"  ✓ easyocr {easyocr.__version__}")
except ImportError as e:
    print(f"  ✗ easyocr: {e}")

print("\nAll critical modules verified!")
PYEOF

# ==================== SUMMARY ====================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}FIX COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${YELLOW}What was done:${NC}"
echo "  ✓ Installed liblzma-dev system library"
echo "  ✓ Rebuilt Python 3.10.13 with lzma support"
echo "  ✓ Reinstalled all Python packages"
echo "  ✓ Verified all imports work"

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Test your application:"
echo -e "   ${GREEN}python hybrid_api.py${NC}"
echo ""
echo "2. If you need to restore packages manually:"
echo -e "   ${GREEN}pip install -r $BACKUP_DIR/requirements.txt${NC}"

echo ""
echo -e "${GREEN}Python is now ready with full lzma support! 🚀${NC}"
echo ""
