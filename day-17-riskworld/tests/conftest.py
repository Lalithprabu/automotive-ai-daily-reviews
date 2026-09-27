import os
import sys

# Make sure `import src...` resolves regardless of the directory pytest is
# invoked from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
