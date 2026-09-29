"""
Plant Health Check — Dataset Loading Utility
Loads the test dataset using Keras ImageDataGenerator.
"""

import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TEST_DIR, IMG_SIZE

def get_test_generator(batch_size=32, shuffle=False):
    """
    Creates and returns a test ImageDataGenerator.
    
    Args:
        batch_size: Batch size for the generator.
        shuffle: Whether to shuffle the data (False for evaluation and metrics).
        
    Returns:
        test_generator: DirectoryIterator yielding batches of (images, labels).
    """
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
        
    # Preprocessing in config.py rescales images to [0, 1]
    datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = datagen.flow_from_directory(
        directory=TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        seed=42
    )
    
    return test_generator

if __name__ == '__main__':
    print("=== Plant Health Check — Dataset Loader ===")
    try:
        generator = get_test_generator(shuffle=False)
        print(f"[OK] Found {generator.samples} samples belonging to {generator.num_classes} classes.")
        print(f"Image shape: {generator.image_shape}")
    except Exception as e:
        print(f"[FAIL] Error loading dataset: {e}")
