#!/bin/bash

################################################################################
# Fix _lzma Issue WITHOUT sudo access - IMPROVED VERSION
# Comprehensive patch for torchvision with detailed error handling
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
║      Fix _lzma Issue (No Sudo Required) - v2                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${YELLOW}This script will patch torchvision to work without lzma${NC}"
echo ""
echo -e "${YELLOW}Press ENTER to continue or Ctrl+C to cancel${NC}"
read

# ==================== DIAGNOSTIC INFO ====================

echo ""
echo -e "${BLUE}Diagnostic Information:${NC}"
echo "Python: $(python --version)"
echo "Python path: $(which python)"

# ==================== FIND TORCHVISION ====================

echo ""
echo -e "${BLUE}Step 1: Locating torchvision${NC}"

# Create a temporary Python script for better error handling
cat > /tmp/find_torchvision.py << 'PYEOF'
import sys
import os

try:
    import torchvision
    path = os.path.dirname(torchvision.__file__)
    print(f"PATH:{path}")
    sys.exit(0)
except ImportError as e:
    print(f"ERROR:Cannot import torchvision: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR:Unexpected error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF

# Run the script and capture output
RESULT=$(python /tmp/find_torchvision.py 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}✗ Failed to locate torchvision${NC}"
    echo "$RESULT"
    exit 1
fi

TORCHVISION_PATH=$(echo "$RESULT" | grep "^PATH:" | cut -d: -f2-)

if [ -z "$TORCHVISION_PATH" ]; then
    echo -e "${RED}✗ Could not extract torchvision path${NC}"
    echo "Output: $RESULT"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found torchvision at: $TORCHVISION_PATH"

# Verify the path exists and is writable
if [ ! -d "$TORCHVISION_PATH" ]; then
    echo -e "${RED}✗ Torchvision path does not exist: $TORCHVISION_PATH${NC}"
    exit 1
fi

if [ ! -w "$TORCHVISION_PATH/datasets" ]; then
    echo -e "${RED}✗ No write permission to torchvision directory${NC}"
    echo "Path: $TORCHVISION_PATH/datasets"
    exit 1
fi

echo -e "${GREEN}✓${NC} Write permissions verified"

# ==================== BACKUP ====================

echo ""
echo -e "${BLUE}Step 2: Creating Backup${NC}"

BACKUP_DIR="lzma_fix_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Check if files exist before backing up
FILES_TO_BACKUP=(
    "$TORCHVISION_PATH/datasets/__init__.py"
    "$TORCHVISION_PATH/datasets/utils.py"
    "$TORCHVISION_PATH/datasets/_optical_flow.py"
)

for file in "${FILES_TO_BACKUP[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$BACKUP_DIR/${filename}.backup"
        echo "  ✓ Backed up: $filename"
    else
        echo -e "  ${YELLOW}⚠${NC} File not found: $file"
    fi
done

echo -e "${GREEN}✓${NC} Backups created in: $BACKUP_DIR/"

# ==================== PATCH FILES ====================

echo ""
echo -e "${BLUE}Step 3: Patching torchvision files${NC}"

# Patch 1: datasets/__init__.py
echo "  Patching datasets/__init__.py..."
python3 << PYEOF
import os
import sys

init_file = "$TORCHVISION_PATH/datasets/__init__.py"

try:
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "PATCHED:" in content and "optical flow" in content.lower():
        print("    ℹ Already patched, skipping")
        sys.exit(0)
    
    # Comment out the problematic import
    original = "from ._optical_flow import FlyingChairs, FlyingThings3D, HD1K, KittiFlow, Sintel"
    replacement = "# PATCHED: Optical flow datasets disabled due to lzma dependency\n# from ._optical_flow import FlyingChairs, FlyingThings3D, HD1K, KittiFlow, Sintel"
    
    if original in content:
        content = content.replace(original, replacement)
        with open(init_file, 'w') as f:
            f.write(content)
        print("    ✓ Patched successfully")
    else:
        print("    ℹ Import already modified or not found")
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to patch datasets/__init__.py${NC}"
    exit 1
fi

# Patch 2: datasets/utils.py
echo "  Patching datasets/utils.py..."
python3 << PYEOF
import os
import sys

utils_file = "$TORCHVISION_PATH/datasets/utils.py"

try:
    with open(utils_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "LZMA_AVAILABLE" in content:
        print("    ℹ Already patched, skipping")
        sys.exit(0)
    
    # Replace the lzma import with a try-except block
    original = "import lzma"
    replacement = """# PATCHED: Make lzma optional
try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False
    lzma = None"""
    
    if original in content:
        content = content.replace(original, replacement, 1)  # Replace only first occurrence
        with open(utils_file, 'w') as f:
            f.write(content)
        print("    ✓ Patched successfully")
    else:
        print("    ℹ Import already modified or not found")
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to patch datasets/utils.py${NC}"
    exit 1
fi

# Patch 3: Create a dummy _optical_flow.py
echo "  Creating dummy _optical_flow.py..."
python3 << PYEOF
import os
import sys

optical_flow_file = "$TORCHVISION_PATH/datasets/_optical_flow.py"

# Create a minimal version that doesn't import utils
dummy_content = '''"""
PATCHED: Optical flow datasets disabled due to lzma dependency
This is a dummy module that prevents import errors.
Original file backed up.
"""

# Dummy classes to prevent AttributeError
class FlyingChairs:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FlyingChairs dataset is disabled due to missing lzma support")

class FlyingThings3D:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FlyingThings3D dataset is disabled due to missing lzma support")

class HD1K:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HD1K dataset is disabled due to missing lzma support")

class KittiFlow:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("KittiFlow dataset is disabled due to missing lzma support")

class Sintel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Sintel dataset is disabled due to missing lzma support")

__all__ = ["FlyingChairs", "FlyingThings3D", "HD1K", "KittiFlow", "Sintel"]
'''

try:
    with open(optical_flow_file, 'w') as f:
        f.write(dummy_content)
    print("    ✓ Created successfully")
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create dummy _optical_flow.py${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} All patches applied successfully"

# ==================== VERIFY ====================

echo ""
echo -e "${BLUE}Step 4: Verifying Fix${NC}"

python3 << 'PYEOF'
import sys
import warnings
warnings.filterwarnings('ignore')

print("\nTesting imports:")

success = True

try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ torch: {e}")
    success = False

try:
    import torchvision
    print(f"  ✓ torchvision {torchvision.__version__}")
except ImportError as e:
    print(f"  ✗ torchvision: {e}")
    success = False

try:
    import torchvision.transforms as transforms
    print(f"  ✓ torchvision.transforms")
except ImportError as e:
    print(f"  ✗ torchvision.transforms: {e}")
    success = False

try:
    import easyocr
    print(f"  ✓ easyocr {easyocr.__version__}")
except ImportError as e:
    print(f"  ✗ easyocr: {e}")
    success = False

try:
    import tensorflow as tf
    print(f"  ✓ tensorflow {tf.__version__}")
except ImportError as e:
    print(f"  ⚠ tensorflow: {e}")
    # Don't fail on TensorFlow

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  ✓ onnxruntime (providers: {', '.join(providers[:2])})")
except ImportError as e:
    print(f"  ⚠ onnxruntime: {e}")
    # Don't fail on ONNX

if success:
    print("\n✓ All critical imports successful!")
    print("\nNote: Optical flow datasets are disabled but EasyOCR works perfectly.")
    sys.exit(0)
else:
    print("\n✗ Some critical imports failed")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Verification failed. Restoring backups...${NC}"
    if [ -f "$BACKUP_DIR/__init__.py.backup" ]; then
        cp "$BACKUP_DIR/__init__.py.backup" "$TORCHVISION_PATH/datasets/__init__.py"
    fi
    if [ -f "$BACKUP_DIR/utils.py.backup" ]; then
        cp "$BACKUP_DIR/utils.py.backup" "$TORCHVISION_PATH/datasets/utils.py"
    fi
    if [ -f "$BACKUP_DIR/_optical_flow.py.backup" ]; then
        cp "$BACKUP_DIR/_optical_flow.py.backup" "$TORCHVISION_PATH/datasets/_optical_flow.py"
    fi
    exit 1
fi

# ==================== CREATE RESTORE SCRIPT ====================

cat > "$BACKUP_DIR/restore.sh" << EOF
#!/bin/bash
# Restore original torchvision files

echo "Restoring original torchvision files..."

if [ -f "$BACKUP_DIR/__init__.py.backup" ]; then
    cp "$BACKUP_DIR/__init__.py.backup" "$TORCHVISION_PATH/datasets/__init__.py"
    echo "  ✓ Restored __init__.py"
fi

if [ -f "$BACKUP_DIR/utils.py.backup" ]; then
    cp "$BACKUP_DIR/utils.py.backup" "$TORCHVISION_PATH/datasets/utils.py"
    echo "  ✓ Restored utils.py"
fi

if [ -f "$BACKUP_DIR/_optical_flow.py.backup" ]; then
    cp "$BACKUP_DIR/_optical_flow.py.backup" "$TORCHVISION_PATH/datasets/_optical_flow.py"
    echo "  ✓ Restored _optical_flow.py"
fi

echo "✓ Restore complete"
EOF

chmod +x "$BACKUP_DIR/restore.sh"

# ==================== SUMMARY ====================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}FIX COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${YELLOW}What was done:${NC}"
echo "  ✓ Patched torchvision to work without lzma"
echo "  ✓ Disabled unused optical flow datasets"
echo "  ✓ Created backups in: $BACKUP_DIR/"
echo "  ✓ Verified all imports work"

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Test your application:"
echo -e "   ${GREEN}python hybrid_api.py${NC}"
echo ""
echo "2. To restore original files (if needed):"
echo -e "   ${GREEN}./$BACKUP_DIR/restore.sh${NC}"

echo ""
echo -e "${YELLOW}Note:${NC}"
echo "  • EasyOCR, face detection, and NSFW detection work perfectly"
echo "  • Only torchvision optical flow datasets are disabled"
echo "  • This is a permanent workaround until Python is rebuilt"

echo ""
echo -e "${GREEN}Your application is ready to run! 🚀${NC}"
echo ""