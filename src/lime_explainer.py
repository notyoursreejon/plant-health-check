"""
LIME (Local Interpretable Model-agnostic Explanations) for Plant Disease Detection.

Generates superpixel-based explanations highlighting which regions of a leaf image
influenced the model's disease classification decision.

Usage:
    python src/lime_explainer.py --image data/test/Tomato___Late_blight/sample.jpg
    python src/lime_explainer.py --image path/to/leaf.jpg --num-samples 2000
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

from src.config import (
    load_model, preprocess_image, load_image_for_display, predict_fn,
    get_prediction_summary, ensure_output_dirs, LIME_OUTPUT_DIR,
    _ensure_class_names, configure_gpu
)


class PlantLIMEExplainer:
    """
    LIME explainer tailored for plant disease classification.

    How LIME works:
    1. Takes the input image and generates ~1000 perturbed versions
       by randomly hiding/showing superpixel regions.
    2. Feeds all perturbed images through the model.
    3. Fits a simple linear model to learn which superpixels
       matter most for the prediction.
    4. Returns weights for each superpixel — positive = supports
       the prediction, negative = contradicts it.
    """

    def __init__(self):
        """Initialize LIME explainer."""
        self.model = load_model()
        _ensure_class_names()
        from src.config import CLASS_NAMES
        self.class_names = CLASS_NAMES
        self.explainer = lime_image.LimeImageExplainer(random_state=42)
        ensure_output_dirs()
        print('[OK] LIME Explainer initialized')

    def _prediction_wrapper(self, images):
        """
        Wrapper for LIME's input format.

        LIME sends images as float64 arrays in [0, 1] range with shape (N, H, W, 3).
        Convert to float32.
        """
        return predict_fn(images)

    def explain(self, img_path, num_samples=1000, top_labels=5):
        """Generate LIME explanation."""
        print(f'   Image: {os.path.basename(img_path)}')
        print(f'   Perturbation samples: {num_samples}')

        img_display = load_image_for_display(img_path)
        predicted_class, confidence, top_results = get_prediction_summary(img_path)

        print(f'   Prediction: {predicted_class} ({confidence:.1%})')
        print('   [WAIT] Generating explanation (this may take 30-120 seconds)...')

        explanation = self.explainer.explain_instance(
            img_display,
            self._prediction_wrapper,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            random_seed=42
        )

        print('   [OK] Explanation generated')
        return (explanation, img_display, predicted_class, confidence, top_results)

    def visualize(self, img_path, num_samples=1000, save=True, show=False):
        """Generate 4-panel visualization."""
        explanation, img_display, predicted_class, confidence, top_results = self.explain(
            img_path, num_samples=num_samples
        )

        predicted_idx = self.class_names.index(predicted_class)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(
            'LIME Explanation — ' + predicted_class.replace('___', ' → ')
            + '  (Confidence: ' + f'{confidence:.1%}' + ')',
            fontsize=14, fontweight='bold', y=1.02
        )

        # Panel 0: Original Image
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image', fontsize=11, fontweight='bold')

        top3 = top_results[:3]
        top3_lines = []
        for i, (name, prob) in enumerate(top3):
            prefix = '>> ' if i == 0 else '   '
            top3_lines.append(prefix + name.replace('___', ' > ') + ': ' + f'{prob:.1%}')
        top3_text = '\n'.join(top3_lines)

        axes[0].text(
            0.02, -0.05, top3_text,
            transform=axes[0].transAxes,
            fontsize=8, fontfamily='monospace', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

        # Panel 1: Positive Regions
        temp_pos, mask_pos = explanation.get_image_and_mask(
            predicted_idx, positive_only=True, num_features=5, hide_rest=False
        )
        axes[1].imshow(mark_boundaries(temp_pos, mask_pos, color=(0, 1, 0), mode='thick'))
        axes[1].set_title('Positive Regions\n(Support Prediction)', fontsize=10, fontweight='bold')

        # Panel 2: Pros & Cons
        temp_pn, mask_pn = explanation.get_image_and_mask(
            predicted_idx, positive_only=False, num_features=10, hide_rest=False
        )
        axes[2].imshow(mark_boundaries(temp_pn, mask_pn, color=(1, 0, 0), mode='thick'))
        axes[2].set_title('Pros & Cons\n(Green=Support, Red=Contradict)', fontsize=10, fontweight='bold')

        # Panel 3: Heatmap
        heatmap = self._generate_heatmap(explanation, predicted_idx, img_display.shape)
        axes[3].imshow(img_display)
        heatmap_overlay = axes[3].imshow(
            heatmap, cmap='RdYlGn', alpha=0.5,
            vmin=-np.max(np.abs(heatmap)), vmax=np.max(np.abs(heatmap))
        )
        plt.colorbar(heatmap_overlay, ax=axes[3], fraction=0.046, pad=0.04, label='Importance')
        axes[3].set_title('Importance Heatmap\n(Green=Positive, Red=Negative)', fontsize=10, fontweight='bold')

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()

        if save:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(LIME_OUTPUT_DIR, img_name + '_lime_explanation.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print('   [SAVED] ' + output_path)

        if show:
            plt.show()

        plt.close(fig)

        if save:
            return output_path
        return None

    def _generate_heatmap(self, explanation, label_idx, img_shape):
        """Convert LIME explanation segments into continuous heatmap."""
        segments = explanation.segments
        weights = dict(explanation.local_exp[label_idx])
        heatmap = np.zeros(img_shape[:2], dtype=np.float64)
        for seg_id, weight in weights.items():
            heatmap[segments == seg_id] = weight
        return heatmap

    def explain_batch(self, img_paths, num_samples=1000):
        """Generate explanations for multiple images."""
        results = []
        for i, path in enumerate(img_paths):
            print(f'\n[{i + 1}/{len(img_paths)}] Processing: {os.path.basename(path)}')
            output = self.visualize(path, num_samples=num_samples, save=True, show=False)
            results.append(output)
            print(f'   [{i + 1}/{len(img_paths)}] Done')
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate LIME explanation for a plant leaf image')
    parser.add_argument('--image', type=str, required=True, help='Path to the leaf image')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of LIME perturbation samples (default: 1000)')
    parser.add_argument('--show', action='store_true', help='Show the plot interactively')
    args = parser.parse_args()
    configure_gpu()
    explainer = PlantLIMEExplainer()
    output = explainer.visualize(args.image, num_samples=args.num_samples, save=True, show=args.show)
    print('\n[DONE] LIME explanation saved to: ' + output)


if __name__ == '__main__':
    main()
