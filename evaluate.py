#!/usr/bin/env python3
"""
Main evaluation script for reviewer agent.
Provides a clean interface to the evaluation system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the evaluation system
from reviewer_agent.eval.evaluation import main

if __name__ == "__main__":
    main()
