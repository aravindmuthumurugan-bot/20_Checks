#!/bin/bash

################################################################################
# Quick Workaround for _lzma Issue (without rebuilding Python)
# This patches torchvision to avoid the lzma import
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
║      Quick Workaround for _lzma Issue                               ║
║      (Does NOT rebuild Python)                                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${YELLOW}This is a TEMPORARY workaround that:${NC}"
echo "  1. Patches torchvision to skip lzma-dependent imports"
echo "  2. Allows your application to run"
echo ""
echo -e "${RED}NOTE: This is NOT a permanent solution${NC}"
echo -e "${YELLOW}For a permanent fix, run: ./fix_lzma_issue.sh${NC}"
echo ""
echo -e "${YELLOW}Press ENTER to continue or Ctrl+C to cancel${NC}"
read

# ==================== FIND TORCHVISION ====================

echo ""
echo -e "${BLUE}Step 1: Locating torchvision${NC}"

TORCHVISION_PATH=$(python -c "import torchvision; import os; print(os.path.dirname(torchvision.__file__))" 2>/dev/null)

if [ -z "$TORCHVISION_PATH" ]; then
    echo -e "${RED}✗ Could not locate torchvision${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found torchvision at: $TORCHVISION_PATH"

# ==================== BACKUP ====================

echo ""
echo -e "${BLUE}Step 2: Creating Backup${NC}"

BACKUP_DIR="lzma_workaround_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp "$TORCHVISION_PATH/datasets/__init__.py" "$BACKUP_DIR/datasets__init__.py.backup"
echo -e "${GREEN}✓${NC} Backup created: $BACKUP_DIR/datasets__init__.py.backup"

# ==================== PATCH TORCHVISION ====================

echo ""
echo -e "${BLUE}Step 3: Patching torchvision${NC}"

cat > /tmp/patch_torchvision.py << 'PYEOF'
import os

torchvision_path = None
try:
    import torchvision
    torchvision_path = os.path.dirname(torchvision.__file__)
except ImportError:
    print("✗ Could not import torchvision")
    exit(1)

init_file = os.path.join(torchvision_path, "datasets", "__init__.py")

# Read current content
with open(init_file, 'r') as f:
    content = f.read()

# Patch: Comment out the problematic import line
new_content = content.replace(
    "from ._optical_flow import FlyingChairs, FlyingThings3D, HD1K, KittiFlow, Sintel",
    "# PATCHED: from ._optical_flow import FlyingChairs, FlyingThings3D, HD1K, KittiFlow, Sintel"
)

# Write patched content
with open(init_file, 'w') as f:
    f.write(new_content)

print("✓ Patched torchvision datasets/__init__.py")
PYEOF

python /tmp/patch_torchvision.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} torchvision patched successfully"
else
    echo -e "${RED}✗ Failed to patch torchvision${NC}"
    exit 1
fi

# ==================== VERIFY ====================

echo ""
echo -e "${BLUE}Step 4: Verifying Fix${NC}"

python3 << 'PYEOF'
import sys

print("\nTesting imports:")

try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ torch: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"  ✓ torchvision {torchvision.__version__}")
except ImportError as e:
    print(f"  ✗ torchvision: {e}")
    sys.exit(1)

try:
    import easyocr
    print(f"  ✓ easyocr {easyocr.__version__}")
except ImportError as e:
    print(f"  ✗ easyocr: {e}")
    sys.exit(1)

print("\n✓ All imports successful!")
print("\nNote: Some torchvision optical flow datasets will not be available,")
print("but this does not affect EasyOCR functionality.")
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Import verification failed${NC}"
    echo "Restoring backup..."
    cp "$BACKUP_DIR/datasets__init__.py.backup" "$TORCHVISION_PATH/datasets/__init__.py"
    exit 1
fi

# ==================== SUMMARY ====================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}WORKAROUND APPLIED!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${YELLOW}What was done:${NC}"
echo "  ✓ Patched torchvision to skip lzma-dependent imports"
echo "  ✓ Created backup at: $BACKUP_DIR/"
echo "  ✓ Verified all imports work"

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Test your application:"
echo -e "   ${GREEN}python hybrid_api.py${NC}"
echo ""
echo "2. For a permanent fix, install liblzma and rebuild Python:"
echo -e "   ${GREEN}./fix_lzma_issue.sh${NC}"
echo ""
echo "3. To restore the original torchvision:"
echo -e "   ${GREEN}cp $BACKUP_DIR/datasets__init__.py.backup $TORCHVISION_PATH/datasets/__init__.py${NC}"

echo ""
echo -e "${YELLOW}⚠ This is a temporary workaround${NC}"
echo -e "${GREEN}Your application should now run! 🚀${NC}"
echo ""
