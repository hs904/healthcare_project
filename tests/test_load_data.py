import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from data.load_data import load_data

# Test the load_data function
df = load_data()
print(df.head())
