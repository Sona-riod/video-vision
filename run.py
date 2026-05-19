#!/usr/bin/env python3
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from main import SimpleKegApp

if __name__ == '__main__':
    # Start the Kivy application
    SimpleKegApp().run()