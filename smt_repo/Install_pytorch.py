import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip3", "install", "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"])