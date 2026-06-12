
import sys

print(f"Python: {sys.version}")

print("\n--- Checking dependencies ---")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import gymnasium
    print(f"✅ Gymnasium: {gymnasium.__version__}")
except Exception as e:
    print(f"❌ Gymnasium: {e}")

try:
    import ale_py
    print(f"✅ ALE-py: {ale_py.__version__}")
except Exception as e:
    print(f"❌ ALE-py: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")

print("\n--- Environment check complete ---")
