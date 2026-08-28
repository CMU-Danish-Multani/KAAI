#!/bin/bash
# Stage A1. Isolated env for LtU-ILI. Kept separate because ltu-ili restricts
# sbi<=0.22.0 while the KAAI env runs sbi 0.27.0. Python 3.11 because sbi 0.22
# predates 3.12 and its wheels may not exist for it.
set -x
CONDA=/Users/danishmultani/miniconda3/bin/conda
$CONDA create -n ltuili python=3.11 -y || exit 1
$CONDA run -n ltuili pip install "sbi<=0.22.0" || echo "SBI PIN FAILED"
$CONDA run -n ltuili pip install "ltu-ili[pytorch]" || \
  $CONDA run -n ltuili pip install "git+https://github.com/maho3/ltu-ili.git"
$CONDA run -n ltuili pip install "tarp @ git+https://github.com/maho3/tarp.git"
$CONDA run -n ltuili python -c "
import ili, torch, sbi
print('OK ili', ili.__version__ if hasattr(ili,'__version__') else '?')
print('OK torch', torch.__version__, 'mps', torch.backends.mps.is_available())
print('OK sbi', sbi.__version__)
try:
    import lampe; print('OK lampe', lampe.__version__)
except Exception as e: print('LAMPE FAILED', e)
try:
    import tarp; print('OK tarp')
except Exception as e: print('TARP FAILED', e)
"
