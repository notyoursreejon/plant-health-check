"""
Plant Health Check — Model Load Test
Quick script to verify that the Keras model loads and runs predictions correctly.
"""

import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
from src.config import load_model, configure_gpu, get_prediction_summary, TEST_DIR

def run_model_test():
    """Verify model loading and run a prediction on a sample test image."""
    print("=== Plant Health Check — Model Load Test ===")
    
    # Configure GPU memory growth if available
    configure_gpu()
    
    # Load the trained model
    try:
        model = load_model()
        print("[OK] Model file verified and loaded into cache.")
    except Exception as e:
        print(f"[FAIL] Error loading model: {e}")
        return
        
    # Find sample images in test directory
    test_images = glob.glob(os.path.join(TEST_DIR, "*", "*.*"))
    if not test_images:
        print("[WARN] No images found in data/test directory. Cannot run prediction test.")
        return
        
    # Run test prediction on the first sample image
    sample_path = test_images[0]
    print(f"\n[INFO] Running prediction test on: {sample_path}")
    
    try:
        pred_class, conf, top_results = get_prediction_summary(sample_path, top_k=3)
        print("\nPrediction Results:")
        print(f"  Predicted Class: {pred_class}")
        print(f"  Confidence:      {conf:.2%}")
        print("  Top 3 classes:")
        for name, prob in top_results:
            print(f"    - {name}: {prob:.2%}")
        print("\n[OK] Model inference is working successfully.")
    except Exception as e:
        print(f"[FAIL] Inference test failed: {e}")

if __name__ == '__main__':
    run_model_test()
