"""
Plant Health Check — Model Evaluation & Pipeline Orchestrator Module
Coordinates benchmarking, metric calculation, table exporting, plot rendering, and report generation.
"""

import os
import json
import shutil
from src.config import PROJECT_ROOT
from src.model_comparator import ModelComparator
from src.visualization_engine import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_learning_curves,
    plot_model_comparison_bar,
    plot_class_wise_performance
)
from src.report_generator import export_all_tables, generate_academic_report


class PipelineEvaluator:
    """
    Main orchestrator for complete end-to-end model evaluation and benchmark reporting.
    """
    def __init__(self):
        self.output_dir = os.path.join(PROJECT_ROOT, "outputs")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        self.tables_dir = os.path.join(self.output_dir, "tables")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        
        # Ensure output directories exist
        for d in [self.metrics_dir, self.tables_dir, self.plots_dir, self.reports_dir]:
            os.makedirs(d, exist_ok=True)
            
    def run(self):
        """Executes full evaluation pipeline."""
        print("=====================================================================")
        print("   PLANT PATHOLOGY AI — IEEE MODEL EVALUATION & BENCHMARK SUITE")
        print("=====================================================================\n")
        
        # 1. Run multi-model benchmark & ablation study
        comparator = ModelComparator(num_classes=39)
        data = comparator.run_benchmark()
        
        val_results = data["val_results"]
        test_results = data["test_results"]
        ablation_results = data["ablation_results"]
        test_labels = data["test_labels"]
        test_probs = data["test_probs"]
        per_class_cnn = data["per_class_cnn"]
        
        # 2. Save JSON Metrics
        with open(os.path.join(self.metrics_dir, "validation_metrics.json"), "w") as f:
            json.dump(val_results, f, indent=2)
            
        with open(os.path.join(self.metrics_dir, "test_metrics.json"), "w") as f:
            json.dump(test_results, f, indent=2)
            
        with open(os.path.join(self.metrics_dir, "ablation_metrics.json"), "w") as f:
            json.dump(ablation_results, f, indent=2)
            
        print(f"[OK] Saved metric JSON files to: {self.metrics_dir}")
        
        # 3. Export Publication Tables
        val_df, test_df, ablation_df, val_md, test_md, ablation_md = export_all_tables(val_results, test_results, ablation_results, self.tables_dir)
        
        # 4. Render High-Resolution Visualizations
        print("\n[INFO] Generating publication-quality 300 DPI figures...")
        pm_preds = test_probs["Proposed PM-CNN"].argmax(axis=1)
        plot_confusion_matrix(test_labels, pm_preds, self.plots_dir)
        plot_roc_curves(test_labels, test_probs, self.plots_dir)
        plot_pr_curves(test_labels, test_probs, self.plots_dir)
        plot_learning_curves(self.plots_dir)
        plot_model_comparison_bar(test_results, self.plots_dir)
        plot_class_wise_performance(per_class_cnn, self.plots_dir)
        
        # 5. Generate Thesis/IEEE Evaluation Report Section
        report_path = os.path.join(self.reports_dir, "evaluation_report.md")
        generate_academic_report(val_df, test_df, ablation_df, report_path)
        
        # Copy to artifact directory if available
        artifact_dir = r"C:\Users\Your Sreejon\.gemini\antigravity\brain\95af8fa3-ad85-4a49-8596-3f179abc70a7"
        if os.path.isdir(artifact_dir):
            shutil.copy(report_path, os.path.join(artifact_dir, "evaluation_report.md"))
            print(f"[OK] Synced evaluation_report.md to artifact directory.")
            
        print("\n=====================================================================")
        print("   EVALUATION COMPLETE — ALL TABLES & PLOTS GENERATED SUCCESSFULLY")
        print("=====================================================================\n")
        
        return {
            "val_results": val_results,
            "test_results": test_results,
            "ablation_results": ablation_results,
            "val_md": val_md,
            "test_md": test_md,
            "ablation_md": ablation_md
        }
