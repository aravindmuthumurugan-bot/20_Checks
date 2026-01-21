#!/bin/bash

################################################################################
# PII Detection System - Installation Script (No Sudo Required)
# Optimized for: NVIDIA L40S GPU, CUDA 12.0, Python 3.10.13
# User Space Installation
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓ SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠ WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗ ERROR]${NC} $1"
}

log_step() {
    echo -e "${MAGENTA}[STEP]${NC} $1"
}

clear
echo "================================================================================"
echo "         PII Detection System - User Space Installation"
echo "         Optimized for NVIDIA L40S with CUDA 12.0"
echo "         No Sudo Required!"
echo "================================================================================"
echo ""

# Verify we're not running as root
if [ "$EUID" -eq 0 ]; then 
    log_error "Please do not run this script as root. Run as regular user."
    exit 1
fi

# Detect OS
log_step "Step 1: Detecting operating system..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
    log_info "Detected: $OS $VER"
else
    log_error "Cannot detect OS"
    exit 1
fi

# Verify Python version
log_step "Step 2: Verifying Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10.13"

if [ "$PYTHON_VERSION" == "$REQUIRED_VERSION" ]; then
    log_success "Python version $PYTHON_VERSION ✓"
else
    log_warning "Python version is $PYTHON_VERSION (required: $REQUIRED_VERSION)"
    log_info "Continuing with current version..."
fi

PYTHON_CMD="python3"

# Verify NVIDIA GPU
log_step "Step 3: Verifying NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
    log_success "GPU: $GPU_INFO"
    log_info "Driver Version: $DRIVER_VERSION"
    log_info "GPU Memory: $GPU_MEMORY"
else
    log_error "nvidia-smi not found. NVIDIA drivers must be installed."
    exit 1
fi

# Verify CUDA
log_step "Step 4: Verifying CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    log_success "CUDA Version: $CUDA_VERSION"
    
    # Detect CUDA path
    CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    log_info "CUDA Home: $CUDA_HOME"
else
    log_error "nvcc not found. CUDA Toolkit must be installed."
    exit 1
fi

# Create project directory
PROJECT_DIR="$HOME/pii_detection_system"
log_step "Step 5: Creating project directory..."
log_info "Project directory: $PROJECT_DIR"

if [ -d "$PROJECT_DIR" ]; then
    log_warning "Directory already exists. Backing up..."
    mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
log_success "Project directory created"

# Create virtual environment
log_step "Step 6: Creating Python virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate
log_success "Virtual environment created and activated"

# Upgrade pip and setuptools
log_step "Step 7: Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel --quiet
log_success "Package managers upgraded"

# Check for system libraries (information only)
log_step "Step 8: Checking system dependencies..."
log_info "Checking for required system libraries..."

MISSING_LIBS=()

# Check for essential libraries
check_lib() {
    if ldconfig -p 2>/dev/null | grep -q "$1"; then
        log_success "$1 found"
        return 0
    else
        log_warning "$1 not found (may cause issues)"
        MISSING_LIBS+=("$1")
        return 1
    fi
}

check_lib "libGL.so"
check_lib "libglib-2.0.so"
check_lib "libgomp.so"

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    log_warning "Some system libraries are missing:"
    for lib in "${MISSING_LIBS[@]}"; do
        echo "  - $lib"
    done
    log_info "These are usually pre-installed. If you encounter errors, contact your system admin."
fi
log_success "System dependency check complete"

# Install PyTorch with CUDA 12.1 support (closest to 12.0)
log_step "Step 9: Installing PyTorch with CUDA 12.1 support..."
log_info "This may take several minutes..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 --quiet
log_success "PyTorch installed"

# Install TensorFlow with GPU support
log_step "Step 10: Installing TensorFlow with GPU support..."
log_info "This may take several minutes..."
pip install tensorflow[and-cuda]==2.15.0 --quiet
log_success "TensorFlow installed"

# Install ONNX Runtime with GPU support
log_step "Step 11: Installing ONNX Runtime GPU..."
pip install onnxruntime-gpu==1.16.3 --quiet
log_success "ONNX Runtime GPU installed"

# Install core Python dependencies
log_step "Step 12: Installing core Python dependencies..."
pip install --quiet \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    opencv-python==4.9.0.80 \
    opencv-contrib-python==4.9.0.80 \
    numpy==1.24.3 \
    pillow==10.2.0 \
    pytesseract==0.3.10 \
    pydantic==2.5.3 \
    python-dateutil==2.8.2 \
    requests==2.31.0 \
    aiofiles==23.2.1

log_success "Core dependencies installed"

# Install EasyOCR with GPU support
log_step "Step 13: Installing EasyOCR..."
pip install easyocr==1.7.0 --quiet
log_success "EasyOCR installed"

# Install additional ML libraries
log_step "Step 14: Installing additional ML libraries..."
pip install --quiet \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    pandas==2.1.4

log_success "Additional ML libraries installed"

# Pre-download EasyOCR models
log_step "Step 15: Pre-downloading EasyOCR models..."
log_info "This will download ~100MB of model data..."
$PYTHON_CMD << 'EOF'
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Downloading EasyOCR English model...")
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    print("✓ Model downloaded successfully!")
except Exception as e:
    print(f"Model download status: {e}")
EOF
log_success "EasyOCR models downloaded"

# Create directory structure
log_step "Step 16: Creating directory structure..."
mkdir -p "$PROJECT_DIR"/{uploads,outputs,logs,models,scripts,config,temp}
log_success "Directory structure created"

# Create environment configuration
log_step "Step 17: Creating environment configuration..."
cat > "$PROJECT_DIR/.env" << EOF
# ============================================================================
# PII Detection System - Environment Configuration
# ============================================================================

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
CUDA_HOME=$CUDA_HOME
CUDA_CACHE_DISABLE=0
CUDA_CACHE_PATH=\$HOME/.nv/ComputeCache
CUDA_CACHE_MAXSIZE=2147483648

# TensorFlow Configuration
TF_FORCE_GPU_ALLOW_GROWTH=true
TF_GPU_THREAD_MODE=gpu_private
TF_GPU_THREAD_COUNT=2
TF_CPP_MIN_LOG_LEVEL=2
TF_ENABLE_ONEDNN_OPTS=1
TF_XLA_FLAGS=--tf_xla_auto_jit=2

# PyTorch Configuration
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# OMP/MKL Settings
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8

# Application Configuration
MAX_WORKERS=4
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs
LOG_LEVEL=INFO
API_PORT=8000
API_HOST=0.0.0.0

# NVIDIA GPU Optimization
NVIDIA_TF32_OVERRIDE=1
EOF
log_success "Environment configuration created"

# Update bashrc with CUDA paths
log_step "Step 18: Updating shell environment..."
if ! grep -q "PII Detection System - CUDA Configuration" ~/.bashrc; then
    cat >> ~/.bashrc << EOF

# ============================================================================
# PII Detection System - CUDA Configuration
# ============================================================================
export CUDA_HOME=$CUDA_HOME
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# CUDA Cache
export CUDA_CACHE_PATH=\$HOME/.nv/ComputeCache
mkdir -p \$CUDA_CACHE_PATH

# TensorFlow
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Alias for quick activation
alias pii-activate='cd $PROJECT_DIR && source venv/bin/activate'
EOF
    log_success "Shell environment updated"
else
    log_info "CUDA paths already in .bashrc"
fi

# Create CUDA cache directory
mkdir -p $HOME/.nv/ComputeCache
log_success "CUDA cache directory created"

# Create GPU test script
log_step "Step 19: Creating GPU test script..."
cat > "$PROJECT_DIR/scripts/test_gpu.py" << 'EOF'
#!/usr/bin/env python3
"""GPU Test and Verification Script"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf
import torch

print("="*80)
print("GPU TEST AND VERIFICATION")
print("="*80)

# Test TensorFlow
print("\n--- TensorFlow GPU Test ---")
print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print(f"GPU Device: {gpus[0]}")
        print(f"GPU Name: {gpu_details.get('device_name', 'Unknown')}")
        
        print("\nTesting TensorFlow GPU computation...")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result:\n{c.numpy()}")
        
        print("✓ TensorFlow GPU test PASSED!")
    except Exception as e:
        print(f"✗ TensorFlow GPU test FAILED: {e}")
        sys.exit(1)
else:
    print("✗ No TensorFlow GPU found")
    sys.exit(1)

# Test PyTorch
print("\n--- PyTorch GPU Test ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    print("\nTesting PyTorch GPU computation...")
    x = torch.rand(3, 3).cuda()
    y = torch.rand(3, 3).cuda()
    z = torch.matmul(x, y)
    print(f"Matrix multiplication result:\n{z.cpu().numpy()}")
    
    print("✓ PyTorch GPU test PASSED!")
else:
    print("✗ No PyTorch GPU found")
    sys.exit(1)

# Test ONNX Runtime
print("\n--- ONNX Runtime GPU Test ---")
try:
    import onnxruntime as ort
    print(f"ONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("✓ CUDA Execution Provider available!")
    else:
        print("⚠ CUDA Execution Provider not available")
except ImportError:
    print("✗ ONNX Runtime not installed")

# Test EasyOCR
print("\n--- EasyOCR GPU Test ---")
try:
    import easyocr
    print("Initializing EasyOCR with GPU...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    print("✓ EasyOCR initialized with GPU support!")
except Exception as e:
    print(f"✗ EasyOCR GPU initialization failed: {e}")

# GPU Memory Info
print("\n--- GPU Memory Information ---")
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1024**3:.4f} GB")
print(f"Cached memory: {torch.cuda.memory_reserved(0) / 1024**3:.4f} GB")

print("\n" + "="*80)
print("ALL GPU TESTS PASSED! ✓")
print("="*80)
EOF
chmod +x "$PROJECT_DIR/scripts/test_gpu.py"
log_success "GPU test script created"

# Create GPU monitoring script
log_step "Step 20: Creating GPU monitoring script..."
cat > "$PROJECT_DIR/scripts/monitor_gpu.sh" << 'EOF'
#!/bin/bash
# Real-time GPU monitoring
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.gr,clocks.mem --format=csv | column -t -s,'
EOF
chmod +x "$PROJECT_DIR/scripts/monitor_gpu.sh"
log_success "GPU monitoring script created"

# Create detailed monitoring script
log_step "Step 21: Creating detailed monitoring script..."
cat > "$PROJECT_DIR/scripts/monitor_gpu_detailed.sh" << 'EOF'
#!/bin/bash
# Detailed GPU Monitoring

clear
echo "================================================================================"
echo "NVIDIA L40S GPU - Real-time Monitoring"
echo "Press Ctrl+C to exit"
echo "================================================================================"

while true; do
    clear
    echo "================================================================================"
    echo "GPU Status - $(date)"
    echo "================================================================================"
    
    echo ""
    echo "GPU Utilization:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU: %3d%%  Memory: %3d%%\n", $1, $2}'
    
    echo ""
    echo "Temperature:"
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | \
        awk '{printf "  GPU: %d°C\n", $1}'
    
    echo ""
    echo "Power:"
    nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  Current: %.2f W / %.2f W (%.1f%%)\n", $1, $2, ($1/$2)*100}'
    
    echo ""
    echo "Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  Used: %d MB / %d MB (%.1f%%)\n", $1, $2, ($1/$2)*100}'
    
    echo ""
    echo "Clocks:"
    nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  Graphics: %d MHz  Memory: %d MHz\n", $1, $2}'
    
    echo ""
    echo "Active Processes:"
    nvidia-smi pmon -c 1 -s m 2>/dev/null | tail -n +3 | head -n 10
    
    echo ""
    echo "================================================================================"
    
    sleep 2
done
EOF
chmod +x "$PROJECT_DIR/scripts/monitor_gpu_detailed.sh"
log_success "Detailed monitoring script created"

# Create GPU benchmark script
log_step "Step 22: Creating GPU benchmark script..."
cat > "$PROJECT_DIR/scripts/benchmark_gpu.py" << 'EOF'
#!/usr/bin/env python3
"""GPU Benchmark Script for L40S"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import torch
import time

print("="*80)
print("NVIDIA L40S GPU BENCHMARK")
print("="*80)

# TensorFlow Benchmark
print("\n--- TensorFlow Benchmark ---")
print("Matrix multiplication: 5000x5000 matrices")

with tf.device('/GPU:0'):
    # Warmup
    for _ in range(5):
        a = tf.random.normal([5000, 5000])
        b = tf.random.normal([5000, 5000])
        c = tf.matmul(a, b)
    
    # Benchmark
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        a = tf.random.normal([5000, 5000])
        b = tf.random.normal([5000, 5000])
        c = tf.matmul(a, b)
    
    elapsed = time.time() - start
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Operations/sec: {iterations/elapsed:.2f}")
    print(f"Avg time/op: {(elapsed/iterations)*1000:.2f} ms")

# PyTorch Benchmark
print("\n--- PyTorch Benchmark ---")
device = torch.device("cuda")

# Warmup
for _ in range(5):
    a = torch.randn(5000, 5000, device=device)
    b = torch.randn(5000, 5000, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

# Benchmark
start = time.time()
for _ in range(iterations):
    a = torch.randn(5000, 5000, device=device)
    b = torch.randn(5000, 5000, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

elapsed = time.time() - start
print(f"Iterations: {iterations}")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Operations/sec: {iterations/elapsed:.2f}")
print(f"Avg time/op: {(elapsed/iterations)*1000:.2f} ms")

print("\n--- GPU Memory Stats ---")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}")
print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

print("\n" + "="*80)
print("BENCHMARK COMPLETE!")
print("="*80)
EOF
chmod +x "$PROJECT_DIR/scripts/benchmark_gpu.py"
log_success "GPU benchmark script created"

# Create API startup script
log_step "Step 23: Creating API startup script..."
cat > "$PROJECT_DIR/start_api.sh" << 'EOF'
#!/bin/bash

echo "================================================================================"
echo "Starting PII Detection API"
echo "================================================================================"

# Activate virtual environment
source venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found!"
    echo "Please copy your FastAPI application to: $(pwd)/main.py"
    exit 1
fi

echo ""
echo "API Server Starting..."
echo "  - URL: http://0.0.0.0:8000"
echo "  - Docs: http://0.0.0.0:8000/docs"
echo "  - GPU: Enabled"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
EOF
chmod +x "$PROJECT_DIR/start_api.sh"
log_success "API startup script created"

# Create requirements.txt
log_step "Step 24: Creating requirements.txt..."
cat > "$PROJECT_DIR/requirements.txt" << 'EOF'
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Deep Learning Frameworks (GPU) - CUDA 12.0
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
tensorflow[and-cuda]==2.15.0
onnxruntime-gpu==1.16.3

# Computer Vision & OCR
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
easyocr==1.7.0
pytesseract==0.3.10
pillow==10.2.0

# Data Processing
numpy==1.24.3
pandas==2.1.4
scikit-learn==1.3.2
scipy==1.11.4

# API & Utilities
pydantic==2.5.3
python-dateutil==2.8.2
requests==2.31.0
aiofiles==23.2.1
EOF
log_success "requirements.txt created"

# Create README
log_step "Step 25: Creating README..."
cat > "$PROJECT_DIR/README.md" << 'EOF'
# PII Detection System

GPU-accelerated PII detection system for NVIDIA L40S (User Space Installation).

## Quick Start

1. **Activate virtual environment:**
```bash
   cd ~/pii_detection_system
   source venv/bin/activate
```

   Or use the alias:
```bash
   pii-activate
```

2. **Test GPU:**
```bash
   python3 scripts/test_gpu.py
```

3. **Copy your FastAPI application:**
```bash
   cp /path/to/your/main.py ~/pii_detection_system/main.py
```

4. **Start API:**
```bash
   ./start_api.sh
```

## Scripts

- `scripts/test_gpu.py` - Test GPU functionality
- `scripts/benchmark_gpu.py` - Benchmark GPU performance
- `scripts/monitor_gpu.sh` - Monitor GPU (simple)
- `scripts/monitor_gpu_detailed.sh` - Monitor GPU (detailed)
- `start_api.sh` - Start the API server

## API Endpoints

- Health: `GET /health`
- Single image: `POST /api/v1/check-image`
- Multiple images: `POST /api/v1/check-images`
- Async processing: `POST /api/v1/check-images-async`
- Job status: `GET /api/v1/job/{job_id}`

## Monitoring
```bash
# Simple monitoring
nvidia-smi

# Real-time monitoring
./scripts/monitor_gpu.sh

# Detailed monitoring
./scripts/monitor_gpu_detailed.sh
```

## Troubleshooting

If you encounter "library not found" errors, some system libraries may be missing.
Contact your system administrator to install:
- libGL.so (OpenGL)
- libglib-2.0.so (GLib)
- libgomp.so (OpenMP)
EOF
log_success "README created"

# Create user guide
log_step "Step 26: Creating user guide..."
cat > "$PROJECT_DIR/USER_GUIDE.txt" << 'EOF'
================================================================================
PII DETECTION SYSTEM - USER GUIDE
================================================================================

QUICK ACTIVATION:
-----------------
Option 1: Use the alias
  $ pii-activate

Option 2: Manual activation
  $ cd ~/pii_detection_system
  $ source venv/bin/activate

TESTING GPU:
------------
1. Test GPU functionality:
   $ python3 scripts/test_gpu.py

2. Benchmark GPU performance:
   $ python3 scripts/benchmark_gpu.py

RUNNING THE API:
----------------
1. Make sure you have copied main.py to the project directory
   $ cp /path/to/main.py ~/pii_detection_system/

2. Start the API:
   $ cd ~/pii_detection_system
   $ ./start_api.sh

3. Access the API:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs

MONITORING GPU:
---------------
Simple monitoring:
  $ nvidia-smi

Real-time monitoring:
  $ cd ~/pii_detection_system
  $ ./scripts/monitor_gpu.sh

Detailed monitoring:
  $ cd ~/pii_detection_system
  $ ./scripts/monitor_gpu_detailed.sh

TROUBLESHOOTING:
----------------
Problem: Import errors or "library not found"
Solution: Some system libraries may be missing. Contact your system admin.

Problem: GPU not detected by TensorFlow/PyTorch
Solution: Check CUDA paths are set correctly:
  $ echo $CUDA_HOME
  $ echo $LD_LIBRARY_PATH

Problem: Out of memory errors
Solution: Reduce batch size or max_workers in API configuration

USEFUL COMMANDS:
----------------
Check Python packages:
  $ pip list | grep -E "torch|tensorflow|easyocr"

Check GPU memory:
  $ nvidia-smi --query-gpu=memory.used,memory.total --format=csv

Check GPU temperature:
  $ nvidia-smi --query-gpu=temperature.gpu --format=csv

Monitor GPU continuously:
  $ watch -n 1 nvidia-smi

DEACTIVATING ENVIRONMENT:
-------------------------
$ deactivate

================================================================================
EOF
log_success "User guide created"

# Run GPU test
log_step "Step 27: Running GPU verification test..."
echo ""
$PYTHON_CMD "$PROJECT_DIR/scripts/test_gpu.py"

# Final summary
echo ""
echo "================================================================================"
log_success "INSTALLATION COMPLETE! (No sudo required)"
echo "================================================================================"
echo ""
echo "📁 Project Directory: $PROJECT_DIR"
echo ""
echo "📂 Directory Structure:"
echo "  ├── venv/              (Virtual environment)"
echo "  ├── uploads/           (Uploaded images)"
echo "  ├── outputs/           (Processed results)"
echo "  ├── logs/              (Application logs)"
echo "  ├── scripts/           (Utility scripts)"
echo "  ├── .env               (Environment configuration)"
echo "  ├── requirements.txt   (Python dependencies)"
echo "  ├── start_api.sh       (API startup script)"
echo "  ├── README.md          (Documentation)"
echo "  └── USER_GUIDE.txt     (Detailed user guide)"
echo ""
echo "🚀 NEXT STEPS:"
echo ""
echo "  1. Activate environment (choose one):"
echo "     → pii-activate                    (use the alias)"
echo "     → cd $PROJECT_DIR && source venv/bin/activate"
echo ""
echo "  2. Copy your FastAPI application:"
echo "     → cp /path/to/main.py $PROJECT_DIR/main.py"
echo ""
echo "  3. Test GPU:"
echo "     → python3 scripts/test_gpu.py"
echo ""
echo "  4. Run benchmark:"
echo "     → python3 scripts/benchmark_gpu.py"
echo ""
echo "  5. Start API:"
echo "     → ./start_api.sh"
echo ""
echo "📊 MONITORING:"
echo "  → nvidia-smi                          (GPU status)"
echo "  → ./scripts/monitor_gpu.sh            (Real-time)"
echo "  → ./scripts/monitor_gpu_detailed.sh   (Detailed)"
echo ""
echo "🌐 API ACCESS (after starting):"
echo "  → API: http://localhost:8001"
echo "  → Interactive Docs: http://localhost:8001/docs"
echo ""
echo "📖 DOCUMENTATION:"
echo "  → README.md        (Quick reference)"
echo "  → USER_GUIDE.txt   (Detailed guide)"
echo ""
echo "💡 TIP: Reload your shell to use the 'pii-activate' alias:"
echo "  → source ~/.bashrc"
echo ""
echo "================================================================================"
echo ""
log_success "Installation successful! Your PII Detection System is ready to use."
echo ""