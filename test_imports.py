#!/usr/bin/env python3
"""Quick test script to verify Python package imports"""

import sys

def test_import(package_name):
    try:
        __import__(package_name)
        print(f"✅ {package_name}: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {package_name}: ERROR - {e}")
        return False

def main():
    print("🐍 Testing Python Package Imports")
    print("=" * 40)
    
    packages = [
        'qonnx',
        'finn', 
        'brevitas',
        'onnx',
        'numpy'
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if test_import(package):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_count} packages imported successfully")
    
    if success_count == total_count:
        print("🎉 All packages imported successfully!")
        return 0
    else:
        print("⚠️ Some packages failed to import")
        return 1

if __name__ == "__main__":
    sys.exit(main())
