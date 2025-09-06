# test_imports.py
# Purpose: make sure core libs are installed and usable.

import sys

print("Python:", sys.version)

try:
    import torch
    print("torch OK:", torch.__version__, "| cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch import FAILED:", repr(e))

try:
    import transformers
    print("transformers OK:", transformers.__version__)
except Exception as e:
    print("transformers import FAILED:", repr(e))

try:
    import accelerate
    print("accelerate OK:", accelerate.__version__)
except Exception as e:
    print("accelerate import FAILED:", repr(e))

try:
    import peft
    print("peft OK:", peft.__version__)
except Exception as e:
    print("peft import FAILED:", repr(e))
