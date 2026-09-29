"""
Plant Health Check — CLI Explainability Runner
Runs LIME and/or SHAP explanations from the command line, with an optional
side-by-side comparison image.

Usage:
    python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg"
    python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg" --lime-only
    python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg" --shap-only
"""

import os
import sys
import argparse
import time

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_comparison(lime_path, shap_path, output_dir):
    """Create a side-by-side comparison image from LIME and SHAP outputs."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("LIME vs SHAP — Explainability Comparison", fontsize=16, fontweight='bold', y=0.98)

    if lime_path and os.path.exists(lime_path):
        lime_img = mpimg.imread(lime_path)
        axes[0].imshow(lime_img)
        axes[0].set_title("LIME Explanation", fontsize=13, fontweight='bold', pad=10)
    else:
        axes[0].text(0.5, 0.5, "LIME not available", ha='center', va='center',
                     fontsize=14, color='gray', transform=axes[0].transAxes)
        axes[0].set_title("LIME Explanation", fontsize=13, color='gray', pad=10)
    axes[0].axis('off')

    if shap_path and os.path.exists(shap_path):
        shap_img = mpimg.imread(shap_path)
        axes[1].imshow(shap_img)
        axes[1].set_title("SHAP Explanation", fontsize=13, fontweight='bold', pad=10)
    else:
        axes[1].text(0.5, 0.5, "SHAP not available", ha='center', va='center',
                     fontsize=14, color='gray', transform=axes[1].transAxes)
        axes[1].set_title("SHAP Explanation", fontsize=13, color='gray', pad=10)
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    comparison_path = os.path.join(output_dir, "comparison.png")
    fig.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return comparison_path


def main():
    parser = argparse.ArgumentParser(
        description="Plant Health Check — CLI Explainability Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg"
  python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg" --lime-only
  python src/explain.py --image "data/test/Tomato___Late_blight/image.jpg" --shap-only
        """
    )
    parser.add_argument("--image", required=True, help="Path to the leaf image to analyze")
    parser.add_argument("--lime-only", action="store_true", help="Run only LIME explanation (~30 seconds)")
    parser.add_argument("--shap-only", action="store_true", help="Run only SHAP explanation (~1-5 minutes)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open output images")
    parser.add_argument("--num-samples", type=int, default=800, help="LIME perturbation samples (default: 800)")
    parser.add_argument("--n-background", type=int, default=25, help="SHAP background images (default: 25)")

    args = parser.parse_args()

    # Resolve image path
    img_path = args.image
    if not os.path.isabs(img_path):
        img_path = os.path.join(PROJECT_ROOT, img_path)

    if not os.path.exists(img_path):
        print(f"\n[ERROR] Image not found: {img_path}")
        sys.exit(1)

    # Determine what to run
    run_lime = not args.shap_only
    run_shap = not args.lime_only

    print("=" * 60)
    print("  Plant Health Check - Explainability Runner")
    print("=" * 60)
    print(f"  Image:  {os.path.basename(img_path)}")
    print(f"  LIME:   {'Yes' if run_lime else 'Skipped'}")
    print(f"  SHAP:   {'Yes' if run_shap else 'Skipped'}")
    print("=" * 60)

    # Step 1: Run prediction
    print("\n[1/3] Running prediction...")
    from src.config import get_prediction_summary, configure_gpu, ensure_output_dirs

    configure_gpu()
    ensure_output_dirs()

    predicted_class, confidence, top_results = get_prediction_summary(img_path, top_k=5)

    print(f"\n  Predicted: {predicted_class}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"\n  Top-5 Predictions:")
    for i, (cls, prob) in enumerate(top_results, 1):
        bar = "#" * int(prob * 30)
        print(f"    {i}. {cls:<45s} {prob:.1%}  {bar}")

    # Step 2: LIME explanation
    lime_path = None
    if run_lime:
        print(f"\n[2/3] Generating LIME explanation ({args.num_samples} samples)...")
        t0 = time.time()
        try:
            from src.lime_explainer import PlantLIMEExplainer
            explainer = PlantLIMEExplainer()
            lime_path = explainer.visualize(img_path, num_samples=args.num_samples, save=True, show=False)
            elapsed = time.time() - t0
            if lime_path:
                print(f"  [OK] LIME complete ({elapsed:.1f}s) -> {lime_path}")
            else:
                print(f"  [ERR] LIME returned no output ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [ERR] LIME failed ({elapsed:.1f}s): {e}")
    else:
        print("\n[2/3] LIME - skipped (--shap-only)")

    # Step 3: SHAP explanation
    shap_path = None
    if run_shap:
        print(f"\n[3/3] Generating SHAP explanation ({args.n_background} background images)...")
        t0 = time.time()
        try:
            from src.shap_explainer import PlantSHAPExplainer
            explainer = PlantSHAPExplainer(n_background=args.n_background)
            shap_path = explainer.visualize(img_path, save=True, show=False)
            elapsed = time.time() - t0
            if shap_path:
                print(f"  [OK] SHAP complete ({elapsed:.1f}s) -> {shap_path}")
            else:
                print(f"  [ERR] SHAP returned no output ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [ERR] SHAP failed ({elapsed:.1f}s): {e}")
    else:
        print("\n[3/3] SHAP - skipped (--lime-only)")

    # Step 4: Comparison (if both ran)
    comparison_path = None
    if lime_path and shap_path:
        print("\n[+] Creating LIME vs SHAP comparison...")
        comparison_dir = os.path.join(PROJECT_ROOT, "outputs", "comparison")
        comparison_path = create_comparison(lime_path, shap_path, comparison_dir)
        print(f"  [OK] Comparison saved -> {comparison_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)
    if lime_path:
        print(f"  LIME:       {lime_path}")
    if shap_path:
        print(f"  SHAP:       {shap_path}")
    if comparison_path:
        print(f"  Comparison: {comparison_path}")
    print("=" * 60)

    # Auto-open images
    if not args.no_open:
        import subprocess
        paths_to_open = [p for p in [lime_path, shap_path, comparison_path] if p]
        if paths_to_open:
            print("\n  Opening output images...")
            for p in paths_to_open:
                try:
                    if sys.platform == "win32":
                        os.startfile(p)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", p], check=False)
                    else:
                        subprocess.run(["xdg-open", p], check=False)
                except Exception:
                    pass

    print("\nDone.")


if __name__ == "__main__":
    main()
