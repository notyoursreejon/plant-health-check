"""
Plant Pathology AI — Master Model Evaluation Runner
Run this script to evaluate all models (CNN Proposed + Classical ML Baselines), compute 12+ metrics,
generate publication-quality tables & 300 DPI plots, and export research paper report chapters.

Usage:
    python main_evaluation.py
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import configure_gpu
from src.evaluator import PipelineEvaluator

if __name__ == '__main__':
    configure_gpu()
    evaluator = PipelineEvaluator()
    evaluator.run()
