"""
SHAP (SHapley Additive exPlanations) for Plant Disease Detection.

Generates Shapley value attribution maps showing how each pixel region
contributes to the model's disease prediction.

Usage:
    python src/shap_explainer.py --image data/test/Tomato___Late_blight/sample.jpg
    python src/shap_explainer.py --image path/to/leaf.jpg --n-background 100
"""

import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
import tensorflow as tf

from src.config import (
    load_model, preprocess_image, load_image_for_display, predict_fn,
    get_prediction_summary, ensure_output_dirs, get_background_images,
    SHAP_OUTPUT_DIR, _ensure_class_names, configure_gpu, IMG_SIZE
)


class PlantSHAPExplainer:
    """
    SHAP explainer for plant disease classification.

    How SHAP works (for images):
    1. Uses a background/reference dataset to establish a "baseline" prediction.
    2. For each pixel region, computes how much adding/removing it changes
       the prediction compared to the baseline.
    3. Assigns each pixel a SHAP value — positive means it pushes toward
       the predicted class, negative means it pushes away.

    The key advantage over LIME: SHAP values are theoretically grounded
    in game theory (Shapley values) and always sum to the difference
    between the model output and the expected output.
    """

    def __init__(self, n_background=50):
        """Initialize SHAP explainer."""
        self.model = load_model()
        _ensure_class_names()
        from src.config import CLASS_NAMES
        self.class_names = CLASS_NAMES
        self.n_background = n_background
        self._explainer = None
        self._explainer_type = None
        ensure_output_dirs()
        print('[OK] SHAP Explainer initialized')

    def _create_explainer(self, background_data):
        """Create best available SHAP explainer with automatic fallback."""
        # Priority: GradientExplainer → DeepExplainer → KernelExplainer
        try:
            print('   [TRY] Trying GradientExplainer...')
            explainer = shap.GradientExplainer(self.model, background_data)
            test_shap = explainer.shap_values(background_data[:1])
            self._explainer_type = 'GradientExplainer'
            print('   [OK] Using GradientExplainer (fast, gradient-based)')
            return explainer
        except Exception as e:
            print('   [WARN] GradientExplainer failed: ' + str(e)[:80])

        try:
            print('   [TRY] Trying DeepExplainer...')
            explainer = shap.DeepExplainer(self.model, background_data)
            test_shap = explainer.shap_values(background_data[:1])
            self._explainer_type = 'DeepExplainer'
            print('   [OK] Using DeepExplainer (DeepLIFT-based)')
            return explainer
        except Exception as e:
            print('   [WARN] DeepExplainer failed: ' + str(e)[:80])

        print('   [TRY] Falling back to KernelExplainer (this will be slow)...')
        print('   [INFO] Downscaling to 64x64 for KernelExplainer feasibility')
        self._explainer_type = 'KernelExplainer'
        return None  # kernel explain is handled separately in _kernel_explain

    def explain(self, img_path):
        """Generate SHAP explanation for a single leaf image."""
        print('\n' + '=' * 60)
        print('SHAP Analysis')
        print('=' * 60)
        print('   [INFO] Image: ' + img_path)

        predicted_class, confidence, top_results = get_prediction_summary(img_path)
        img_preprocessed = preprocess_image(img_path)
        img_display = load_image_for_display(img_path)

        print('   [INFO] Loading background dataset (' + str(self.n_background) + ' images)...')
        background = get_background_images(n_samples=self.n_background)

        if self._explainer is None:
            self._explainer = self._create_explainer(background)

        print('   [WAIT] Computing SHAP values (this may take 1-5 minutes)...')

        if self._explainer_type == 'KernelExplainer':
            shap_values = self._kernel_explain(img_preprocessed, background)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shap_values = self._explainer.shap_values(img_preprocessed)

        print('   [OK] SHAP values computed')
        return (shap_values, img_display, predicted_class, confidence, top_results)

    def _kernel_explain(self, img, background):
        """KernelExplainer fallback on 64x64 downscaled images, then upscale back to 224x224."""
        from skimage.transform import resize

        img_small = resize(img[0], (64, 64, 3), anti_aliasing=True)
        bg_small = np.array([resize(b, (64, 64, 3), anti_aliasing=True) for b in background])

        img_flat = img_small.reshape(1, -1)
        bg_flat = bg_small.reshape(bg_small.shape[0], -1)

        def predict_flat(flat_imgs):
            imgs = flat_imgs.reshape(-1, 64, 64, 3)
            resized = np.array([resize(im, (224, 224, 3), anti_aliasing=True) for im in imgs])
            return predict_fn(resized)

        explainer = shap.KernelExplainer(predict_flat, bg_flat)
        sv_flat = explainer.shap_values(img_flat, nsamples=100)

        n_classes = len(self.class_names)
        shap_values = []
        for c in range(n_classes):
            sv_small = sv_flat[c].reshape(64, 64, 3)
            sv_full = resize(sv_small, (224, 224, 3), anti_aliasing=True)
            shap_values.append(np.expand_dims(sv_full, 0))

        return shap_values

    def visualize(self, img_path, save=True, show=False):
        """Generate 3-panel SHAP explanation."""
        shap_values, img_display, predicted_class, confidence, top_results = self.explain(img_path)

        predicted_idx = self.class_names.index(predicted_class)

        # Handle shap_values format
        if isinstance(shap_values, list):
            sv_predicted = shap_values[predicted_idx]
        else:
            if shap_values.ndim == 5:
                sv_predicted = shap_values[..., predicted_idx]
            else:
                sv_predicted = shap_values

        # Compute heatmap (sum across channels)
        sv_heatmap = np.sum(np.abs(sv_predicted[0]), axis=-1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            'SHAP Explanation — ' + predicted_class.replace('___', ' → ') +
            '  (Confidence: ' + f'{confidence:.1%}' + ')  [' +
            self._explainer_type + ']',
            fontsize=14, fontweight='bold', y=1.02
        )

        # Panel 0: Original Image
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Top-3 text box
        top3_text = '\n'.join([
            f'{i+1}. {name}: {conf:.1%}'
            for i, (name, conf) in enumerate(top_results[:3])
        ])
        axes[0].text(
            0.02, 0.98, top3_text,
            transform=axes[0].transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

        # Panel 1: SHAP Heatmap
        shap_cmap = LinearSegmentedColormap.from_list('shap', ('#0000FF', '#FFFFFF', '#FF0000'))
        sv_signed = np.sum(sv_predicted[0], axis=-1)
        max_abs = np.max(np.abs(sv_signed)) + 1e-8

        axes[1].imshow(img_display)
        hm = axes[1].imshow(sv_signed, cmap=shap_cmap, alpha=0.6, vmin=-max_abs, vmax=max_abs)
        plt.colorbar(hm, ax=axes[1], label='SHAP Value')
        axes[1].set_title(
            "SHAP Heatmap\n(Red=Supports '" + predicted_class.split('___')[-1] + "', Blue=Opposes)",
            fontsize=12, fontweight='bold'
        )
        axes[1].axis('off')

        # Panel 2: Top-5 Class Bar Chart
        mean_shap_per_class = []
        for c_idx in range(len(self.class_names)):
            if isinstance(shap_values, list) and c_idx < len(shap_values):
                sv_c = shap_values[c_idx]
            else:
                sv_c = shap_values[..., c_idx]
            mean_shap_per_class.append(np.mean(np.abs(sv_c)))
        mean_shap_per_class = np.array(mean_shap_per_class)

        top5_idx = np.argsort(mean_shap_per_class)[::-1][:5]
        colors = ['#e74c3c' if i == top5_idx[0] else '#3498db' for i in top5_idx]
        class_labels = [self.class_names[i].replace('___', '\n→ ')[:30] for i in top5_idx]
        values = mean_shap_per_class[top5_idx]

        axes[2].barh(range(len(top5_idx)), values, color=colors, edgecolor='white', linewidth=2)
        axes[2].set_yticks(range(len(top5_idx)))
        axes[2].set_yticklabels(class_labels)
        x_offset = max(values) * 0.02 if max(values) > 0 else 1e-6
        for i, v in enumerate(values):
            axes[2].text(v + x_offset, i, f'{v:.4f}', va='center', ha='left', fontsize=9)
        axes[2].set_xlabel('Mean |SHAP Value|')
        axes[2].set_title('Top-5 Class Attributions', fontsize=12, fontweight='bold')
        axes[2].invert_yaxis()

        plt.tight_layout()

        output_path = None
        if save:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(SHAP_OUTPUT_DIR, img_name + '_shap_explanation.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print('   [SAVED] ' + output_path)

        if show:
            plt.show()

        plt.close(fig)

        return output_path if save else None

    def explain_batch(self, img_paths):
        """Generate SHAP explanations for multiple images."""
        for i, img_path in enumerate(img_paths):
            print(f'\n[{i+1}/{len(img_paths)}] Processing: {img_path}')
            self.visualize(img_path, save=True)


def main():
    parser = argparse.ArgumentParser(description='Generate SHAP explanation for a plant leaf image')
    parser.add_argument('--image', type=str, required=True, help='Path to the leaf image')
    parser.add_argument('--n-background', type=int, default=50, help='Number of background images (default: 50)')
    parser.add_argument('--show', action='store_true', help='Show the plot interactively')
    args = parser.parse_args()
    configure_gpu()
    explainer = PlantSHAPExplainer(n_background=args.n_background)
    output = explainer.visualize(args.image, save=True, show=args.show)
    print('\n[DONE] SHAP explanation saved to: ' + output)


if __name__ == '__main__':
    main()
