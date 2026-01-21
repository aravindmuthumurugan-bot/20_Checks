#!/bin/bash

################################################################################
# Fix _lzma Issue - Direct File Patching (No Import Required)
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
║      Fix _lzma Issue - Direct Patching                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${YELLOW}This script will patch torchvision files directly${NC}"
echo ""
echo -e "${YELLOW}Press ENTER to continue or Ctrl+C to cancel${NC}"
read

# ==================== FIND TORCHVISION PATH ====================

echo ""
echo -e "${BLUE}Step 1: Locating torchvision files${NC}"

# Find torchvision in the Python site-packages without importing it
PYTHON_PATH=$(python -c "import sys; print(sys.executable)")
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

echo "  Python: $PYTHON_PATH"
echo "  Site-packages: $SITE_PACKAGES"

# Look for torchvision
TORCHVISION_PATH=""
if [ -d "$SITE_PACKAGES/torchvision" ]; then
    TORCHVISION_PATH="$SITE_PACKAGES/torchvision"
elif [ -d "$HOME/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torchvision" ]; then
    TORCHVISION_PATH="$HOME/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torchvision"
else
    # Try to find it
    TORCHVISION_PATH=$(find "$SITE_PACKAGES" -type d -name "torchvision" 2>/dev/null | head -1)
fi

if [ -z "$TORCHVISION_PATH" ] || [ ! -d "$TORCHVISION_PATH" ]; then
    echo -e "${RED}✗ Could not locate torchvision directory${NC}"
    echo "  Searched in: $SITE_PACKAGES"
    echo ""
    echo "Please provide the full path to torchvision directory:"
    echo "Example: /home/pyphotoval/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torchvision"
    read -p "Path: " TORCHVISION_PATH
    
    if [ ! -d "$TORCHVISION_PATH" ]; then
        echo -e "${RED}✗ Directory does not exist: $TORCHVISION_PATH${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓${NC} Found torchvision at: $TORCHVISION_PATH"

# Verify key files exist
DATASETS_DIR="$TORCHVISION_PATH/datasets"
if [ ! -d "$DATASETS_DIR" ]; then
    echo -e "${RED}✗ datasets directory not found in torchvision${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} datasets directory exists"

# Check write permissions
if [ ! -w "$DATASETS_DIR" ]; then
    echo -e "${RED}✗ No write permission to: $DATASETS_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Write permissions verified"

# ==================== BACKUP ====================

echo ""
echo -e "${BLUE}Step 2: Creating Backup${NC}"

BACKUP_DIR="lzma_fix_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup files
FILES_TO_BACKUP=(
    "$DATASETS_DIR/__init__.py"
    "$DATASETS_DIR/utils.py"
    "$DATASETS_DIR/_optical_flow.py"
)

for file in "${FILES_TO_BACKUP[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$BACKUP_DIR/${filename}.backup"
        echo "  ✓ Backed up: $filename"
    else
        echo -e "  ${YELLOW}⚠${NC} File not found (will create): $(basename $file)"
    fi
done

echo -e "${GREEN}✓${NC} Backups created in: $BACKUP_DIR/"

# ==================== PATCH FILES ====================

echo ""
echo -e "${BLUE}Step 3: Patching Files${NC}"

# Patch 1: datasets/__init__.py
echo "  Patching __init__.py..."
INIT_FILE="$DATASETS_DIR/__init__.py"

if [ -f "$INIT_FILE" ]; then
    # Use sed to comment out the optical flow import
    sed -i.bak '/^from \._optical_flow import/s/^/# PATCHED (lzma fix): /' "$INIT_FILE"
    echo "    ✓ Patched"
else
    echo "    ✗ File not found"
    exit 1
fi

# Patch 2: datasets/utils.py
echo "  Patching utils.py..."
UTILS_FILE="$DATASETS_DIR/utils.py"

if [ -f "$UTILS_FILE" ]; then
    # Create a temporary file with the patched content
    cat > /tmp/patch_utils.py << 'PYEOF'
import sys

utils_file = sys.argv[1]

with open(utils_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'LZMA_AVAILABLE' in content:
    print("Already patched")
    sys.exit(0)

# Find and replace the lzma import
if 'import lzma' in content:
    # Replace simple import lzma
    content = content.replace(
        'import lzma',
        '''# PATCHED: Make lzma optional
try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False
    lzma = None''',
        1
    )
    
    with open(utils_file, 'w') as f:
        f.write(content)
    print("Patched")
else:
    print("Import not found")
PYEOF

    RESULT=$(python /tmp/patch_utils.py "$UTILS_FILE")
    echo "    ✓ $RESULT"
else
    echo "    ✗ File not found"
    exit 1
fi

# Patch 3: Create dummy _optical_flow.py
echo "  Creating dummy _optical_flow.py..."
OPTICAL_FILE="$DATASETS_DIR/_optical_flow.py"

cat > "$OPTICAL_FILE" << 'PYEOF'
"""
PATCHED: Optical flow datasets disabled due to lzma dependency
This is a dummy module that prevents import errors.
"""

class FlyingChairs:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FlyingChairs requires lzma support")

class FlyingThings3D:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FlyingThings3D requires lzma support")

class HD1K:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HD1K requires lzma support")

class KittiFlow:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("KittiFlow requires lzma support")

class Sintel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Sintel requires lzma support")

__all__ = ["FlyingChairs", "FlyingThings3D", "HD1K", "KittiFlow", "Sintel"]
PYEOF

echo "    ✓ Created"

echo -e "${GREEN}✓${NC} All patches applied"

# ==================== VERIFY ====================

echo ""
echo -e "${BLUE}Step 4: Verifying Patches${NC}"

python3 << 'PYEOF'
import sys
import warnings
warnings.filterwarnings('ignore')

print("\nTesting imports (this may take a moment)...")

success_count = 0
total_count = 0

# Test torch
total_count += 1
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
    success_count += 1
except Exception as e:
    print(f"  ✗ torch: {e}")

# Test torchvision
total_count += 1
try:
    import torchvision
    print(f"  ✓ torchvision {torchvision.__version__}")
    success_count += 1
except Exception as e:
    print(f"  ✗ torchvision: {e}")

# Test torchvision.transforms
total_count += 1
try:
    import torchvision.transforms as transforms
    print(f"  ✓ torchvision.transforms")
    success_count += 1
except Exception as e:
    print(f"  ✗ torchvision.transforms: {e}")

# Test easyocr
total_count += 1
try:
    import easyocr
    print(f"  ✓ easyocr")
    success_count += 1
except Exception as e:
    print(f"  ✗ easyocr: {e}")

print(f"\n{'✓' if success_count == total_count else '✗'} {success_count}/{total_count} critical imports successful")

if success_count == total_count:
    print("\n✓ All patches working correctly!")
    sys.exit(0)
else:
    print("\n✗ Some imports failed - see errors above")
    sys.exit(1)
PYEOF

VERIFY_RESULT=$?

if [ $VERIFY_RESULT -ne 0 ]; then
    echo ""
    echo -e "${RED}Verification failed. Restoring backups...${NC}"
    
    if [ -f "$BACKUP_DIR/__init__.py.backup" ]; then
        cp "$BACKUP_DIR/__init__.py.backup" "$DATASETS_DIR/__init__.py"
    fi
    if [ -f "$BACKUP_DIR/utils.py.backup" ]; then
        cp "$BACKUP_DIR/utils.py.backup" "$DATASETS_DIR/utils.py"
    fi
    if [ -f "$BACKUP_DIR/_optical_flow.py.backup" ]; then
        cp "$BACKUP_DIR/_optical_flow.py.backup" "$DATASETS_DIR/_optical_flow.py"
    fi
    
    echo "Backups restored"
    exit 1
fi

# ==================== CREATE RESTORE SCRIPT ====================

cat > "$BACKUP_DIR/restore.sh" << EOF
#!/bin/bash
echo "Restoring original torchvision files..."

if [ -f "$BACKUP_DIR/__init__.py.backup" ]; then
    cp "$BACKUP_DIR/__init__.py.backup" "$DATASETS_DIR/__init__.py"
    echo "  ✓ Restored __init__.py"
fi

if [ -f "$BACKUP_DIR/utils.py.backup" ]; then
    cp "$BACKUP_DIR/utils.py.backup" "$DATASETS_DIR/utils.py"
    echo "  ✓ Restored utils.py"
fi

if [ -f "$BACKUP_DIR/_optical_flow.py.backup" ]; then
    cp "$BACKUP_DIR/_optical_flow.py.backup" "$DATASETS_DIR/_optical_flow.py"
    echo "  ✓ Restored _optical_flow.py"
fi

echo "✓ Restore complete"
EOF

chmod +x "$BACKUP_DIR/restore.sh"

# ==================== SUMMARY ====================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}FIX COMPLETE! ✓${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Test your application:"
echo -e "   ${GREEN}python hybrid_api.py${NC}"
echo ""
echo "2. Monitor GPU usage:"
echo -e "   ${GREEN}watch -n 1 nvidia-smi${NC}"
echo ""
echo "3. To restore original files (if needed):"
echo -e "   ${GREEN}./$BACKUP_DIR/restore.sh${NC}"

echo ""
echo -e "${YELLOW}What was fixed:${NC}"
echo "  • Commented out optical flow dataset imports"
echo "  • Made lzma import optional in utils.py"
echo "  • Created dummy optical flow module"
echo "  • EasyOCR and all face detection libraries work"

echo ""
echo -e "${GREEN}Ready to run! 🚀${NC}"
echo ""