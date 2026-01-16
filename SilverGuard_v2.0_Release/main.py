# SilverGuard/main.py
import sys
import os

# Ensure current directory is in path (especially if run from outside)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.engine import SilverGuardEngine

def main():
    engine = SilverGuardEngine()
    engine.run()

if __name__ == '__main__':
    main()