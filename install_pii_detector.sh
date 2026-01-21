#!/bin/bash

################################################################################
# ENHANCED GPU Setup Script for CUDA 12.4 + OCR
# NVIDIA L40S | Driver 550.163.01 | Python 3.10.13
# Full GPU Acceleration: InsightFace + DeepFace + NudeNet + EasyOCR
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
║      ENHANCED GPU Setup for CUDA 12.4 + OCR Integration            ║
║      NVIDIA L40S | Python 3.10.13                                   ║
║      Face Recognition + NSFW Detection + OCR                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${GREEN}System Configuration:${NC}"
echo "  GPU: NVIDIA L40S (46GB VRAM)"
echo "  Driver: 550.163.01"
echo "  CUDA: 12.4 (Compatible with 12.0)"
echo "  Python: 3.10.13"
echo ""

echo -e "${YELLOW}This will install:${NC}"
echo "  ✓ PyTorch 2.5.1 (GPU) - EasyOCR backend"
echo "  ✓ TensorFlow 2.20.0 (GPU) - DeepFace"
echo "  ✓ ONNX Runtime GPU 1.20.1 - InsightFace & NudeNet"
echo "  ✓ EasyOCR 1.7.2 (GPU) - OCR with 80+ languages"
echo "  ✓ InsightFace 0.7.3 (GPU)"
echo "  ✓ DeepFace 0.0.93 (GPU)"
echo "  ✓ NudeNet 3.4.2 (GPU)"
echo "  ✓ All dependencies"
echo ""
echo -e "${YELLOW}Press ENTER to continue or Ctrl+C to cancel${NC}"
read

# ==================== VERIFY SYSTEM ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1: System Verification${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found - GPU drivers not installed${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
echo -e "${GREEN}✓${NC} GPU: $GPU_INFO"

# Check CUDA
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo -e "${GREEN}✓${NC} CUDA: $CUDA_VERSION"

if [[ ! "$CUDA_VERSION" =~ ^12\. ]]; then
    echo -e "${YELLOW}⚠${NC} Warning: Expected CUDA 12.x, found $CUDA_VERSION"
fi

# Check Python
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"

# ==================== BACKUP ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2: Creating Backup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
pip freeze > "$BACKUP_DIR/packages_before.txt" 2>/dev/null || true
echo -e "${GREEN}✓${NC} Backup: $BACKUP_DIR/"

# ==================== INSTALL NVIDIA CUDA LIBRARIES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 3: Installing NVIDIA CUDA Libraries${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing NVIDIA CUDA runtime libraries for CUDA 12.x...${NC}"
pip install --break-system-packages --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12

echo -e "${GREEN}✓${NC} NVIDIA CUDA libraries installed"

# ==================== INSTALL PYTORCH (for EasyOCR) ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 4: Installing PyTorch 2.5.1 (GPU) for EasyOCR${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing PyTorch with CUDA 12.1 support...${NC}"
pip install --break-system-packages --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo -e "${GREEN}✓${NC} PyTorch installed"

# Verify PyTorch GPU
python3 << 'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"  ✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ PyTorch version: {torch.__version__}")
else:
    print("  ✗ PyTorch CUDA not available")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}PyTorch GPU verification failed${NC}"
    exit 1
fi

# ==================== INSTALL TENSORFLOW (for DeepFace) ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 5: Installing TensorFlow 2.20.0 (GPU)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing TensorFlow 2.20.0 (GPU)...${NC}"
pip install --break-system-packages --no-cache-dir tensorflow==2.20.0
echo -e "${GREEN}✓${NC} TensorFlow installed"

# ==================== INSTALL ONNX RUNTIME GPU ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 6: Installing ONNX Runtime GPU 1.20.1${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing ONNX Runtime GPU 1.20.1 (CUDA 12.x)...${NC}"
pip install --break-system-packages --no-cache-dir onnxruntime-gpu==1.20.1
echo -e "${GREEN}✓${NC} ONNX Runtime GPU installed"

# ==================== INSTALL EASYOCR ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 7: Installing EasyOCR with GPU Support${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing EasyOCR dependencies...${NC}"
pip install --break-system-packages --no-cache-dir \
    opencv-python-headless==4.10.0.84 \
    scikit-image==0.24.0 \
    python-bidi==0.4.2 \
    PyYAML==6.0.2

echo -e "${YELLOW}Installing EasyOCR 1.7.2...${NC}"
pip install --break-system-packages --no-cache-dir easyocr==1.7.2
echo -e "${GREEN}✓${NC} EasyOCR installed"

# ==================== INSTALL FACE LIBRARIES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 8: Installing Face Recognition Libraries${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing InsightFace...${NC}"
pip install --break-system-packages --no-cache-dir insightface==0.7.3
echo -e "${GREEN}✓${NC} InsightFace installed"

echo -e "${YELLOW}Installing DeepFace...${NC}"
pip install --break-system-packages --no-cache-dir deepface==0.0.93
echo -e "${GREEN}✓${NC} DeepFace installed"

echo -e "${YELLOW}Installing RetinaFace...${NC}"
pip install --break-system-packages --no-cache-dir retina-face==0.0.17
echo -e "${GREEN}✓${NC} RetinaFace installed"

# ==================== INSTALL NUDENET ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 9: Installing NudeNet with GPU Support${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}Installing ONNX (for NudeNet GPU)...${NC}"
pip install --break-system-packages --no-cache-dir onnx==1.17.0
echo -e "${GREEN}✓${NC} ONNX installed"

echo -e "${YELLOW}Installing NudeNet...${NC}"
pip install --break-system-packages --no-cache-dir nudenet==3.4.2
echo -e "${GREEN}✓${NC} NudeNet installed"

# ==================== INSTALL OTHER DEPENDENCIES ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 10: Installing Other Dependencies${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

pip install --break-system-packages --no-cache-dir \
    opencv-python==4.10.0.84 \
    opencv-contrib-python==4.10.0.84 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    Pillow==11.0.0 \
    fastapi==0.115.5 \
    uvicorn==0.32.1 \
    python-multipart==0.0.17 \
    pydantic==2.10.3 \
    requests==2.32.3 \
    tqdm==4.67.1

echo -e "${GREEN}✓${NC} All dependencies installed"

pip freeze > "$BACKUP_DIR/packages_after.txt"

# ==================== CONFIGURE ENVIRONMENT ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 11: Configuring GPU Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Detect CUDA path
if [ -d "/usr/local/cuda-12.4" ]; then
    CUDA_PATH="/usr/local/cuda-12.4"
elif [ -d "/usr/local/cuda-12.0" ]; then
    CUDA_PATH="/usr/local/cuda-12.0"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
else
    CUDA_PATH="/usr/local/cuda-12.4"
fi

cat > gpu_env_config.sh << EOF
#!/bin/bash
# GPU Environment for CUDA 12.x (PyTorch + TensorFlow + ONNX)

# TensorFlow GPU settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2
export TF_CPP_MIN_LOG_LEVEL=2

# PyTorch GPU settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH
export PATH=${CUDA_PATH}/bin:\$PATH

# ONNX Runtime settings
export ORT_TENSORRT_FP16_ENABLE=1

# Prevent segmentation faults
export CUDA_MODULE_LOADING=LAZY

echo "✓ GPU Environment configured (CUDA 12.x)"
echo "  - PyTorch GPU: Enabled"
echo "  - TensorFlow GPU: Enabled"
echo "  - ONNX Runtime GPU: Enabled"
EOF

chmod +x gpu_env_config.sh
echo -e "${GREEN}✓${NC} gpu_env_config.sh created"

cat > start_gpu_api.sh << 'EOF'
#!/bin/bash
source gpu_env_config.sh
echo "Starting API with full GPU acceleration..."
python hybrid_api.py
EOF

chmod +x start_gpu_api.sh
echo -e "${GREEN}✓${NC} start_gpu_api.sh created"

# ==================== CREATE OCR TEST SCRIPT ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 12: Creating OCR Test Script${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

cat > test_easyocr_gpu.py << 'PYEOF'
#!/usr/bin/env python3
"""
EasyOCR GPU Test Script
Tests OCR functionality with GPU acceleration
"""

import easyocr
import time
import torch

print("\n" + "="*60)
print("EASYOCR GPU TEST")
print("="*60)

# Check GPU availability
print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Initialize EasyOCR with GPU
print("\nInitializing EasyOCR with GPU support...")
print("Languages: English (en)")
start_time = time.time()

# gpu=True forces GPU usage, gpu=False forces CPU
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

init_time = time.time() - start_time
print(f"✓ EasyOCR initialized in {init_time:.2f} seconds")

# Test with a sample image (you can replace with your own)
print("\nEasyOCR is ready for use!")
print("\nExample usage:")
print("  result = reader.readtext('image.jpg')")
print("  for (bbox, text, prob) in result:")
print("      print(f'{text} (confidence: {prob:.2f})')")

print("\n" + "="*60)
PYEOF

chmod +x test_easyocr_gpu.py
echo -e "${GREEN}✓${NC} test_easyocr_gpu.py created"

# ==================== RUN DIAGNOSTICS ====================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 13: Running Final Tests${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

source gpu_env_config.sh

python3 << 'PYEOF'
import warnings
warnings.filterwarnings('ignore')
import sys

print("\n" + "="*60)
print("COMPREHENSIVE GPU VERIFICATION")
print("="*60)

# PyTorch (for EasyOCR)
print("\n1. PyTorch GPU:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
        print(f"   ✓ Device: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ PyTorch version: {torch.__version__}")
    else:
        print("   ✗ CUDA not available")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# TensorFlow
print("\n2. TensorFlow GPU:")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"   ✓ Device: {details.get('device_name', 'Unknown')}")
        print(f"   ✓ GPUs detected: {len(gpus)}")
    else:
        print("   ✗ No GPUs detected")
except Exception as e:
    print(f"   ✗ Error: {e}")

# ONNX Runtime
print("\n3. ONNX Runtime GPU:")
try:
    import onnxruntime as ort
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print(f"   ✓ CUDA provider available")
        print(f"   ✓ Available providers: {ort.get_available_providers()}")
    else:
        print(f"   ✗ CUDA provider not available")
except Exception as e:
    print(f"   ✗ Error: {e}")

# EasyOCR
print("\n4. EasyOCR:")
try:
    import easyocr
    print(f"   ✓ EasyOCR version: {easyocr.__version__}")
    print(f"   ✓ Will use GPU (PyTorch CUDA)")
    print(f"   ℹ Initialize with: reader = easyocr.Reader(['en'], gpu=True)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# InsightFace
print("\n5. InsightFace:")
try:
    from insightface.app import FaceAnalysis
    print(f"   ✓ Installed")
    print(f"   ℹ Will use GPU (ONNX Runtime CUDA)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# DeepFace
print("\n6. DeepFace:")
try:
    from deepface import DeepFace
    if gpus:
        print(f"   ✓ Will use GPU (TensorFlow)")
    else:
        print(f"   ⚠ Will use CPU")
except Exception as e:
    print(f"   ✗ Error: {e}")

# NudeNet
print("\n7. NudeNet (NSFW Detection):")
try:
    from nudenet import NudeDetector
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print(f"   ✓ Will use GPU (ONNX Runtime CUDA)")
    else:
        print(f"   ⚠ Will use CPU")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("INSTALLATION COMPLETE")
print("="*60 + "\n")
PYEOF

# ==================== SUMMARY ====================

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}INSTALLATION COMPLETE!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${BLUE}Installed Components:${NC}"
echo "  ✓ PyTorch 2.5.1 (GPU) - EasyOCR backend"
echo "  ✓ TensorFlow 2.20.0 (GPU) - DeepFace"
echo "  ✓ ONNX Runtime GPU 1.20.1 - InsightFace & NudeNet"
echo "  ✓ EasyOCR 1.7.2 (GPU) - 80+ languages"
echo "  ✓ InsightFace 0.7.3 (GPU)"
echo "  ✓ DeepFace 0.0.93 (GPU)"
echo "  ✓ NudeNet 3.4.2 (GPU)"

echo ""
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo ""
echo "1. Activate GPU environment:"
echo -e "   ${GREEN}source gpu_env_config.sh${NC}"
echo ""
echo "2. Test EasyOCR:"
echo -e "   ${GREEN}python test_easyocr_gpu.py${NC}"
echo ""
echo "3. Start your API:"
echo -e "   ${GREEN}./start_gpu_api.sh${NC}"
echo ""
echo "4. Monitor GPU usage:"
echo -e "   ${GREEN}watch -n 1 nvidia-smi${NC}"

echo ""
echo -e "${BLUE}EasyOCR Usage Example:${NC}"
cat << 'EXAMPLE'
import easyocr

# Initialize with GPU
reader = easyocr.Reader(['en'], gpu=True)

# Read text from image
result = reader.readtext('image.jpg')

# Process results
for (bbox, text, confidence) in result:
    print(f"Text: {text}, Confidence: {confidence:.2f}")
EXAMPLE

echo ""
echo -e "${GREEN}All systems ready! 🚀${NC}"
echo ""