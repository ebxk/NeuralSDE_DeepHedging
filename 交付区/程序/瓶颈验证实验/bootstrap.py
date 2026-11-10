"""Import frozen predecessors without editing them."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
for folder in ('完整认购', '创新组合实验', '扩展组合实验'):
    sys.path.insert(0, str(HERE.parent / folder))

