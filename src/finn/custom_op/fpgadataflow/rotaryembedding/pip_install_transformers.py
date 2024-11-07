def install_transformers():
    import subprocess
    import sys

    try:
        print("transformers not found, installing transformers package via pip")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    except subprocess.CalledProcessError:
        print("Error installing transformers package via pip")
        print("Please install transformers package manually using 'pip install transformers'")

def import_transformers():
    try:
        import transformers
    except ImportError:
        print("transformers not found, installing transformers package via pip")
        install_transformers()

import_transformers()