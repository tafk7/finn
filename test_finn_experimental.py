#!/usr/bin/env python3
"""Test finn-experimental import variants"""

import sys
import os

def test_import_variant(import_name):
    try:
        module = __import__(import_name)
        print(f"✅ {import_name}: SUCCESS")
        print(f"   Module: {module}")
        print(f"   File: {getattr(module, '__file__', 'N/A')}")
        print(f"   Path: {getattr(module, '__path__', 'N/A')}")
        return True
    except ImportError as e:
        print(f"❌ {import_name}: FAILED - {e}")
        return False

def main():
    print("🧪 Testing finn-experimental import variants")
    print("=" * 50)
    
    variants = [
        'finn_experimental',
        'finn-experimental',
        'finnexperimental',
        'experimental'
    ]
    
    for variant in variants:
        test_import_variant(variant)
        print()
    
    print("📁 Available site-packages:")
    try:
        import site
        for path in site.getsitepackages():
            if os.path.exists(path):
                print(f"  {path}")
                experimental_files = [f for f in os.listdir(path) if 'finn' in f.lower() or 'experimental' in f.lower()]
                for f in experimental_files:
                    print(f"    - {f}")
    except Exception as e:
        print(f"Error listing site-packages: {e}")

if __name__ == "__main__":
    main()
