"""
Shared configuration and utilities for plant disease model explainability.
Provides model loading, image preprocessing, class names, and prediction wrappers.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# --- Path Constants ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'leaf_disease_model.h5')
TEST_DIR = os.path.join(PROJECT_ROOT, 'data', 'test')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
LIME_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'lime')
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'shap')
COMPARISON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'comparison')

# --- Image Constants ---
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)

# --- Module-level Variables ---
CLASS_NAMES = None
_model_cache = None


def configure_gpu():
    """Configure GPU memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print('[OK] GPU detected: ' + str(len(gpus)) + ' device(s) - ' + str([gpu.name for gpu in gpus]))
        except RuntimeError as e:
            print('[WARN] GPU config error: ' + str(e))
    else:
        print('[INFO] No GPU detected - running on CPU (explanations will be slower)')


def get_class_names():
    """Auto-discover class names from test directory subfolders, sorted alphabetically."""
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError('Test directory not found: ' + TEST_DIR)
    class_names = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    if not class_names:
        raise ValueError('No class subdirectories found in: ' + TEST_DIR)
    print('[INFO] Found ' + str(len(class_names)) + ' classes')
    return class_names


def _ensure_class_names():
    """Sets global CLASS_NAMES if None by calling get_class_names()."""
    global CLASS_NAMES
    if CLASS_NAMES is None:
        CLASS_NAMES = get_class_names()


def load_model():
    """Load the trained Keras model (cached after first call)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError('Model file not found: ' + MODEL_PATH)
    print('[INFO] Loading model from: ' + MODEL_PATH)
    _model_cache = tf.keras.models.load_model(MODEL_PATH)
    print('[OK] Model loaded successfully')
    print('  Input shape:  ' + str(_model_cache.input_shape))
    print('  Output shape: ' + str(_model_cache.output_shape))
    print('  Parameters:   ' + f'{_model_cache.count_params():,}')
    return _model_cache


def preprocess_image(img_path):
    """Load and preprocess a single image.

    Returns:
        img_batch: numpy array of shape (1, 224, 224, 3), float32, scaled to [0,1].
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError('Image file not found: ' + img_path)
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def load_image_for_display(img_path):
    """Load image as displayable numpy array (0-1 float, no batch dim)."""
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    return keras_image.img_to_array(img) / 255.0


def predict_fn(images):
    """Prediction wrapper compatible with LIME and SHAP.

    LIME sends batches of perturbed images as float64 arrays.

    Args:
        images: numpy array of images.

    Returns:
        predictions: numpy array of shape (N, num_classes).
    """
    model = load_model()
    images = np.array(images, dtype=np.float32)
    if images.max() > 1.0:
        images = images / 255.0
    predictions = model.predict(images, verbose=0)
    return predictions


def get_prediction_summary(img_path, top_k=5):
    """Get formatted prediction summary.

    Args:
        img_path: path to the image file.
        top_k: number of top predictions to return.

    Returns:
        tuple: (predicted_class, confidence, top_results)
            - predicted_class: string name of the top predicted class.
            - confidence: float confidence score.
            - top_results: list of (class_name, probability) tuples.
    """
    _ensure_class_names()
    img_batch = preprocess_image(img_path)
    predictions = predict_fn(img_batch)
    probs = predictions[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_results = [(CLASS_NAMES[i], float(probs[i])) for i in top_indices]
    predicted_class = CLASS_NAMES[np.argmax(probs)]
    confidence = float(np.max(probs))
    return (predicted_class, confidence, top_results)


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    for d in [OUTPUT_DIR, LIME_OUTPUT_DIR, SHAP_OUTPUT_DIR, COMPARISON_OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)


def get_background_images(n_samples, seed=42):
    """Collect random sample of preprocessed images from test set for SHAP.

    Args:
        n_samples: number of background images to collect.
        seed: random seed for reproducibility.

    Returns:
        background: numpy array of shape (n_samples, 224, 224, 3), float32.
    """
    np.random.seed(seed)
    all_images = []
    for class_dir in os.listdir(TEST_DIR):
        class_path = os.path.join(TEST_DIR, class_dir)
        if os.path.isdir(class_path):
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    all_images.append(os.path.join(class_path, fname))
    if len(all_images) < n_samples:
        print('[WARN] Only ' + str(len(all_images)) + ' images available, requested ' + str(n_samples))
        n_samples = len(all_images)
    selected = np.random.choice(all_images, size=n_samples, replace=False)
    background = np.zeros((n_samples, 224, 224, 3), dtype=np.float32)
    for i, path in enumerate(selected):
        background[i] = preprocess_image(path)[0]
    print('[INFO] Background dataset: ' + str(n_samples) + ' images loaded')
    return background
