#!/usr/bin/env python3
import sys
import os

print("Current working directory:", os.getcwd())
print("Python path before modification:", sys.path)

# Try the original approach
print("\n=== Testing original approach ===")
sys.path_original = sys.path.copy()
if '..' not in sys.path:
    sys.path.insert(0, '..')
print("Python path after adding '..':", sys.path)

# Test if we can find the datasets directory
for path in sys.path:
    datasets_path = os.path.join(path, 'datasets')
    if os.path.exists(datasets_path):
        print(f"Found datasets directory at: {datasets_path}")
        if os.path.exists(os.path.join(datasets_path, 'import_dataset.py')):
            print(f"Found import_dataset.py at: {os.path.join(datasets_path, 'import_dataset.py')}")
        else:
            print(f"import_dataset.py NOT found at: {os.path.join(datasets_path, 'import_dataset.py')}")
    else:
        print(f"datasets directory NOT found at: {datasets_path}")

# Reset path and try the corrected approach
print("\n=== Testing corrected approach ===")
sys.path = sys.path_original.copy()
current_dir = os.getcwd()
pieclam_root = os.path.dirname(current_dir)
print(f"Current directory: {current_dir}")
print(f"PieClam root: {pieclam_root}")
if pieclam_root not in sys.path:
    sys.path.insert(0, pieclam_root)
print("Python path after adding pieclam_root:", sys.path)

# Test if we can find the datasets directory
for path in sys.path:
    datasets_path = os.path.join(path, 'datasets')
    if os.path.exists(datasets_path):
        print(f"Found datasets directory at: {datasets_path}")
        if os.path.exists(os.path.join(datasets_path, 'import_dataset.py')):
            print(f"Found import_dataset.py at: {os.path.join(datasets_path, 'import_dataset.py')}")
        else:
            print(f"import_dataset.py NOT found at: {os.path.join(datasets_path, 'import_dataset.py')}")
    else:
        print(f"datasets directory NOT found at: {datasets_path}")

# Try to import
print("\n=== Testing import ===")
try:
    from datasets.import_dataset import import_dataset
    print("SUCCESS: Import worked!")
except ImportError as e:
    print(f"FAILED: Import error: {e}")
except Exception as e:
    print(f"FAILED: Other error: {e}") 