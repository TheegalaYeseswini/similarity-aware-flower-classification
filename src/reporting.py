from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_metrics_from_confusion_matrix(confusion_path: Path) -> Dict[str, float]:
    matrix = np.load(confusion_path)
    matrix = matrix.astype(np.float64)
    total = matrix.sum()
    true_positives = np.diag(matrix)
    precision = np.divide(
        true_positives,
        matrix.sum(axis=0),
        out=np.zeros_like(true_positives),
        where=matrix.sum(axis=0) != 0,
    )
    recall = np.divide(
        true_positives,
        matrix.sum(axis=1),
        out=np.zeros_like(true_positives),
        where=matrix.sum(axis=1) != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positives),
        where=(precision + recall) != 0,
    )
    weights = np.divide(
        matrix.sum(axis=1),
        total,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=total != 0,
    )
    return {
        "test_accuracy": float(true_positives.sum() / total),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * weights).sum()),
    }


def load_result_summaries() -> Dict[str, Dict[str, float]]:
    ce_metrics = _compute_metrics_from_confusion_matrix(
        PROJECT_ROOT / "checkpoints" / "test_confusion_matrix.npy"
    )

    adaptive_payload = _safe_read_json(
        PROJECT_ROOT / "checkpoints_adaptive_loss" / "test_metrics.json"
    )
    similarity_payload = _safe_read_json(
        PROJECT_ROOT / "checkpoints_similarity_loss" / "final_metrics.json"
    )

    return {
        "cross_entropy": {
            "validation_accuracy": None,
            "test_accuracy": ce_metrics["test_accuracy"],
            "macro_precision": ce_metrics["macro_precision"],
            "macro_recall": ce_metrics["macro_recall"],
            "macro_f1": ce_metrics["macro_f1"],
            "weighted_f1": ce_metrics["weighted_f1"],
        },
        "entropy_aware": {
            "validation_accuracy": adaptive_payload["validation_metrics"]["accuracy"],
            "test_accuracy": adaptive_payload["test_metrics"]["accuracy"],
            "macro_precision": adaptive_payload["test_metrics"]["precision_macro"],
            "macro_recall": adaptive_payload["test_metrics"]["recall_macro"],
            "macro_f1": adaptive_payload["test_metrics"]["f1_macro"],
            "weighted_f1": adaptive_payload["test_metrics"]["f1_weighted"],
        },
        "similarity_aware": {
            "validation_accuracy": similarity_payload["validation_metrics"]["accuracy"],
            "test_accuracy": similarity_payload["test_metrics"]["accuracy"],
            "macro_precision": similarity_payload["test_metrics"]["precision_macro"],
            "macro_recall": similarity_payload["test_metrics"]["recall_macro"],
            "macro_f1": similarity_payload["test_metrics"]["f1_macro"],
            "weighted_f1": similarity_payload["test_metrics"]["f1_weighted"],
        },
    }


def render_markdown_results_table() -> str:
    summaries = load_result_summaries()

    def fmt(value):
        if value is None:
            return "Not archived"
        return f"{100.0 * value:.2f}%"

    rows = [
        "| Method | Validation Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Cross Entropy baseline | {fmt(summaries['cross_entropy']['validation_accuracy'])} "
            f"| {fmt(summaries['cross_entropy']['test_accuracy'])} "
            f"| {fmt(summaries['cross_entropy']['macro_precision'])} "
            f"| {fmt(summaries['cross_entropy']['macro_recall'])} "
            f"| {fmt(summaries['cross_entropy']['macro_f1'])} "
            f"| {fmt(summaries['cross_entropy']['weighted_f1'])} |"
        ),
        (
            f"| Adaptive entropy-aware loss | {fmt(summaries['entropy_aware']['validation_accuracy'])} "
            f"| {fmt(summaries['entropy_aware']['test_accuracy'])} "
            f"| {fmt(summaries['entropy_aware']['macro_precision'])} "
            f"| {fmt(summaries['entropy_aware']['macro_recall'])} "
            f"| {fmt(summaries['entropy_aware']['macro_f1'])} "
            f"| {fmt(summaries['entropy_aware']['weighted_f1'])} |"
        ),
        (
            f"| Similarity-aware adaptive loss | {fmt(summaries['similarity_aware']['validation_accuracy'])} "
            f"| {fmt(summaries['similarity_aware']['test_accuracy'])} "
            f"| {fmt(summaries['similarity_aware']['macro_precision'])} "
            f"| {fmt(summaries['similarity_aware']['macro_recall'])} "
            f"| {fmt(summaries['similarity_aware']['macro_f1'])} "
            f"| {fmt(summaries['similarity_aware']['weighted_f1'])} |"
        ),
    ]
    return "\n".join(rows)
