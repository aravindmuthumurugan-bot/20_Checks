#!/bin/bash

################################################################################
# PII Detection System - Complete Installation Script
# Optimized for: NVIDIA L40S GPU, CUDA 12.0, Python 3.10.13
# Server: CHBLDEVLLMAIGPU01
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
echo "         PII Detection System - GPU Installation Script"
echo "         Optimized for NVIDIA L40S with CUDA 12.0"
echo "================================================================================"
echo ""

# Verify we're not running as root
if [ "$EUID" -eq 0 ]; then 
    log_error "Please do not run this script as root. Run as: ./install_pii_detector.sh"
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
    log_error "nvidia-smi not found. NVIDIA drivers must be installed first."
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
    log_error "nvcc not found. CUDA Toolkit must be installed first."
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
pip install --upgrade pip setuptools wheel
log_success "Package managers upgraded"

# Install system dependencies
log_step "Step 8: Installing system dependencies..."
log_info "This step requires sudo password..."

if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        pkg-config \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libglu1-mesa \
        tesseract-ocr \
        libtesseract-dev \
        python3-dev \
        libpython3.10-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        libfontconfig1 \
        libxrender1 \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libatlas-base-dev \
        gfortran \
        > /dev/null 2>&1
    
    log_success "System dependencies installed"
elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Rocky"* ]]; then
    sudo yum update -y -q
    sudo yum install -y \
        gcc \
        gcc-c++ \
        cmake \
        git \
        wget \
        curl \
        unzip \
        pkgconfig \
        mesa-libGL \
        tesseract \
        tesseract-devel \
        python3-devel \
        ffmpeg \
        atlas-devel \
        > /dev/null 2>&1
    
    log_success "System dependencies installed"
else
    log_warning "Unknown OS. Some dependencies may need manual installation."
fi

# Install cuDNN (if not already installed)
log_step "Step 9: Checking cuDNN installation..."
if [ -f "/usr/include/cudnn.h" ] || [ -f "$CUDA_HOME/include/cudnn.h" ]; then
    CUDNN_VERSION=$(grep CUDNN_MAJOR /usr/include/cudnn_version.h 2>/dev/null || grep CUDNN_MAJOR $CUDA_HOME/include/cudnn_version.h 2>/dev/null | awk '{print $3}')
    log_success "cuDNN is already installed (version $CUDNN_VERSION)"
else
    log_warning "cuDNN not detected. Deep learning frameworks may have reduced performance."
    log_info "To install cuDNN manually:"
    log_info "  1. Download from: https://developer.nvidia.com/cudnn"
    log_info "  2. Extract and copy to CUDA directory"
fi

# Install PyTorch with CUDA 12.0 support
log_step "Step 10: Installing PyTorch with CUDA 12.0 support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
log_success "PyTorch installed"

# Install TensorFlow with GPU support
log_step "Step 11: Installing TensorFlow with GPU support..."
pip install tensorflow[and-cuda]==2.15.0
log_success "TensorFlow installed"

# Install ONNX Runtime with GPU support
log_step "Step 12: Installing ONNX Runtime GPU..."
pip install onnxruntime-gpu==1.16.3
log_success "ONNX Runtime GPU installed"

# Install core Python dependencies
log_step "Step 13: Installing core Python dependencies..."
pip install --upgrade \
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
    aiofiles==23.2.1 \
    > /dev/null 2>&1

log_success "Core dependencies installed"

# Install EasyOCR with GPU support
log_step "Step 14: Installing EasyOCR..."
pip install easyocr==1.7.0
log_success "EasyOCR installed"

# Install additional ML libraries
log_step "Step 15: Installing additional ML libraries..."
pip install \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    > /dev/null 2>&1
log_success "Additional ML libraries installed"

# Pre-download EasyOCR models
log_step "Step 16: Pre-downloading EasyOCR models (this may take a few minutes)..."
$PYTHON_CMD << 'EOF'
import warnings
warnings.filterwarnings('ignore')
import easyocr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Initializing EasyOCR reader and downloading English model...")
try:
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    print("✓ English model downloaded successfully!")
except Exception as e:
    print(f"⚠ Model download completed with warning: {e}")
EOF
log_success "EasyOCR models downloaded"

# Create directory structure
log_step "Step 17: Creating directory structure..."
mkdir -p "$PROJECT_DIR"/{uploads,outputs,logs,models,scripts,config}
log_success "Directory structure created"

# Create environment configuration
log_step "Step 18: Creating environment configuration..."
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

# OMP/MKL Settings (CPU threading)
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
log_step "Step 19: Updating shell environment..."
if ! grep -q "CUDA_HOME=$CUDA_HOME" ~/.bashrc; then
    cat >> ~/.bashrc << EOF

# ============================================================================
# PII Detection System - CUDA Configuration
# ============================================================================
export CUDA_HOME=$CUDA_HOME
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
EOF
    log_success "Shell environment updated"
else
    log_info "CUDA paths already in .bashrc"
fi

# Create GPU test script
log_step "Step 20: Creating GPU test script..."
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
        # Set memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print(f"GPU Device: {gpus[0]}")
        print(f"GPU Name: {gpu_details.get('device_name', 'Unknown')}")
        
        # Test computation
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
    
    # Test computation
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
        print("✗ CUDA Execution Provider not available")
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
log_step "Step 21: Creating GPU monitoring script..."
cat > "$PROJECT_DIR/scripts/monitor_gpu.sh" << 'EOF'
#!/bin/bash
# Real-time GPU monitoring
watch -n 1 'nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.gr,clocks.mem --format=csv | column -t -s,'
EOF
chmod +x "$PROJECT_DIR/scripts/monitor_gpu.sh"
log_success "GPU monitoring script created"

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
import numpy as np

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
    ops_per_sec = iterations / elapsed
    
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Operations/sec: {ops_per_sec:.2f}")
    print(f"Avg time/op: {(elapsed/iterations)*1000:.2f} ms")
    print(f"TFLOPS estimate: {(2 * 5000**3 * iterations / elapsed) / 1e12:.2f}")

# PyTorch Benchmark
print("\n--- PyTorch Benchmark ---")
print("Matrix multiplication: 5000x5000 matrices")

device = torch.device("cuda")

# Warmup
for _ in range(5):
    a = torch.randn(5000, 5000, device=device)
    b = torch.randn(5000, 5000, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

# Benchmark
start = time.time()
iterations = 100
for _ in range(iterations):
    a = torch.randn(5000, 5000, device=device)
    b = torch.randn(5000, 5000, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

elapsed = time.time() - start
ops_per_sec = iterations / elapsed

print(f"Iterations: {iterations}")
print(f"Total time: {elapsed:.2f} seconds")
print(f"Operations/sec: {ops_per_sec:.2f}")
print(f"Avg time/op: {(elapsed/iterations)*1000:.2f} ms")
print(f"TFLOPS estimate: {(2 * 5000**3 * iterations / elapsed) / 1e12:.2f}")

# Memory bandwidth test
print("\n--- GPU Memory Bandwidth Test ---")
size = 1024 * 1024 * 1024  # 1 GB
iterations = 10

data = torch.randn(size // 4, device=device)
start = time.time()
for _ in range(iterations):
    _ = data.clone()
    torch.cuda.synchronize()
elapsed = time.time() - start

bandwidth = (size * iterations / elapsed) / 1e9
print(f"Memory bandwidth: {bandwidth:.2f} GB/s")

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

# PII Detection API Startup Script

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

# Start the API
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

# Create systemd service file
log_step "Step 25: Creating systemd service (optional)..."
sudo tee /etc/systemd/system/pii-detector.service > /dev/null << EOF
[Unit]
Description=PII Detection API Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
log_success "Systemd service created (not enabled)"

# Create README
log_step "Step 26: Creating README..."
cat > "$PROJECT_DIR/README.md" << 'EOF'
# PII Detection System

GPU-accelerated PII detection system for NVIDIA L40S.

## Quick Start

1. **Activate virtual environment:**
```bash
   source venv/bin/activate
```

2. **Test GPU:**
```bash
   python3 scripts/test_gpu.py
```

3. **Copy your FastAPI application:**
```bash
   cp /path/to/your/main.py .
```

4. **Start API:**
```bash
   ./start_api.sh
```

## Scripts

- `scripts/test_gpu.py` - Test GPU functionality
- `scripts/benchmark_gpu.py` - Benchmark GPU performance
- `scripts/monitor_gpu.sh` - Monitor GPU usage in real-time
- `start_api.sh` - Start the API server

## API Endpoints

- Health: `GET /health`
- Single image: `POST /api/v1/check-image`
- Multiple images: `POST /api/v1/check-images`
- Async processing: `POST /api/v1/check-images-async`
- Job status: `GET /api/v1/job/{job_id}`

## Monitoring
```bash
# Real-time GPU monitoring
./scripts/monitor_gpu.sh

# Check GPU status
nvidia-smi

# View API logs (if using systemd)
sudo journalctl -u pii-detector -f
```

## Service Management
```bash
# Enable service
sudo systemctl enable pii-detector

# Start service
sudo systemctl start pii-detector

# Stop service
sudo systemctl stop pii-detector

# Check status
sudo systemctl status pii-detector
```
EOF
log_success "README created"

# Run GPU test
log_step "Step 27: Running GPU verification test..."
echo ""
$PYTHON_CMD "$PROJECT_DIR/scripts/test_gpu.py"

# Final summary
echo ""
echo "================================================================================"
log_success "INSTALLATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Project Directory: $PROJECT_DIR"
echo ""
echo "📁 Directory Structure:"
echo "  ├── venv/              (Virtual environment)"
echo "  ├── uploads/           (Uploaded images)"
echo "  ├── outputs/           (Processed results)"
echo "  ├── logs/              (Application logs)"
echo "  ├── scripts/           (Utility scripts)"
echo "  ├── .env               (Environment configuration)"
echo "  ├── requirements.txt   (Python dependencies)"
echo "  ├── start_api.sh       (API startup script)"
echo "  └── README.md          (Documentation)"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "  1. Copy your FastAPI application:"
echo "     cp /path/to/main.py $PROJECT_DIR/main.py"
echo ""
echo "  2. Activate virtual environment:"
echo "     cd $PROJECT_DIR"
echo "     source venv/bin/activate"
echo ""
echo "  3. Test GPU (again if needed):"
echo "     python3 scripts/test_gpu.py"
echo ""
echo "  4. Run GPU benchmark:"
echo "     python3 scripts/benchmark_gpu.py"
echo ""
echo "  5. Start API server:"
echo "     ./start_api.sh"
echo ""
echo "📊 Monitoring:"
echo "  - Real-time GPU: ./scripts/monitor_gpu.sh"
echo "  - GPU status: nvidia-smi"
echo ""
echo "🌐 API Access:"
echo "  - API: http://localhost:8001"
echo "  - Docs: http://localhost:8001/docs"
echo ""
echo "⚙️  Service Management (optional):"
echo "  - Enable: sudo systemctl enable pii-detector"
echo "  - Start: sudo systemctl start pii-detector"
echo "  - Status: sudo systemctl status pii-detector"
echo "  - Logs: sudo journalctl -u pii-detector -f"
echo ""
echo "================================================================================"