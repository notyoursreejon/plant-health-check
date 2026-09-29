"""
Plant Health Check — Standard CNN Baseline Model
Wrapper for the pre-trained MobileNetV2 architecture used as the standard deep learning baseline.
"""

import os
import tensorflow as tf
from src.config import MODEL_PATH, load_model


def get_standard_cnn():
    """
    Returns the loaded Standard CNN Baseline model (MobileNetV2).
    
    Returns:
        model: tf.keras.Model
    """
    return load_model()


def get_standard_cnn_summary():
    """
    Returns model parameter complexity metadata for Standard CNN Baseline.
    
    Returns:
        dict: Parameter counts and model description.
    """
    model = get_standard_cnn()
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        "name": "Standard CNN (MobileNetV2)",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "input_shape": (224, 224, 3),
        "num_classes": 39,
        "description": "Standard MobileNetV2 depthwise-separable architecture with dense classification head."
    }
