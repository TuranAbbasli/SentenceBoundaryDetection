"""
Evaluation functionality for text segmenters.
"""
import json
from tqdm import tqdm
from typing import List, Union, Optional, TYPE_CHECKING

import sklearn.metrics

from azcharboundary.utils.features import PositionLabels
from azcharboundary.utils.types import MetricsResult

if TYPE_CHECKING:
    from azcharboundary.segmenter import TextSegmenter


class Evaluator:
    """
    Handles evaluation of text segmenters.
    """

    @staticmethod
    def evaluate(
        segmenter: "TextSegmenter",
        data: Union[str, List[str]],
        max_samples: Optional[int] = None,
        sample_rate: float = 0.1,
        misclassified_path: Optional[str] = "misclassified.jsonl",
    ) -> MetricsResult:
        """
        Evaluate the model on a dataset and save misclassified cases.

        Args:
            segmenter: The TextSegmenter to evaluate.
            data: List of annotated texts or path to a file.
            max_samples: Maximum number of samples to process.
            sample_rate: Sampling rate for feature extraction.
            misclassified_path: Output JSONL file for saving misclassified examples.

        Returns:
            MetricsResult: Evaluation metrics.
        """
        if not segmenter.is_trained:
            raise ValueError("Model has not been trained yet.")

        all_true_labels = []
        all_predictions = []

        # Define misclassified list
        misclassified = []

        for i, text in enumerate(tqdm(data, total=max_samples or len(data))):
            if max_samples is not None and i >= max_samples:
                break
            Evaluator._evaluate_text(
                segmenter, 
                text, 
                all_true_labels, 
                all_predictions, 
                misclassified,
                sample_rate,
            )

        # Generate evaluation report
        report: MetricsResult = {
            "accuracy": 0.0,
            "boundary_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "binary_mode": True,
        }

        if all_true_labels and all_predictions:
            # Calculate metrics
            accuracy = sklearn.metrics.accuracy_score(all_true_labels, all_predictions)

            # Classification report
            class_report = sklearn.metrics.classification_report(
                y_true=all_true_labels,
                y_pred=all_predictions,
                output_dict=True,
                zero_division=0,
            )

            # Extract metrics for the boundary class (1)
            if "1" in class_report:
                precision = class_report["1"]["precision"]
                recall = class_report["1"]["recall"]
                f1_score = class_report["1"]["f1-score"]
            else:
                precision = 0.0
                recall = 0.0
                f1_score = 0.0

            # Update report
            report["accuracy"] = accuracy
            report["precision"] = precision
            report["recall"] = recall
            report["f1_score"] = f1_score

            # Also calculate boundary-specific metrics (on positions where either true or predicted is a boundary)
            boundary_indices = [
                i
                for i, (t, p) in enumerate(zip(all_true_labels, all_predictions))
                if t == 1 or p == 1
            ]

            if boundary_indices:
                boundary_true = [all_true_labels[i] for i in boundary_indices]
                boundary_pred = [all_predictions[i] for i in boundary_indices]
                boundary_accuracy = sklearn.metrics.accuracy_score(
                    boundary_true, boundary_pred
                )
                report["boundary_accuracy"] = boundary_accuracy

        # Save misclassified cases 
        if misclassified_path:
            with open(misclassified_path, "w", encoding="utf-8") as f:
                for entry in misclassified:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return report

    @staticmethod
    def _evaluate_text(
        segmenter: "TextSegmenter",
        text: str,
        all_true_labels: PositionLabels,
        all_predictions: PositionLabels,
        misclassified: list, 
        sample_rate: float = 0.1,
    ) -> None:
        """
        Evaluate a single annotated text, collecting true labels, predictions,
        and misclassified cases.

        Args:
            segmenter: The TextSegmenter to use.
            text: Annotated text input.
            all_true_labels: Accumulated true labels.
            all_predictions: Accumulated predictions.
            misclassified: List where misclassified samples will be appended.
            sample_rate: Fraction of positions to sample for evaluation.
        """
        features, true_labels = (
            segmenter.feature_extractor.process_annotated_text(
                text,
                segmenter.config.left_window,
                segmenter.config.right_window,
                sample_rate=sample_rate
            )
        )

        if len(features) == 0:
            print(f"DEBUG! Text: {text}\n")
            print(f"DEBUG! Features: {features}\n")
            print(f"DEBUG! Labels: {true_labels}\n")
        else:
            predictions = segmenter.model.predict(features)

            # Add all predictions and labels for proper evaluation
            all_true_labels.extend(true_labels)
            all_predictions.extend(predictions)

            # Collect misclassified cases
            for idx, (true, pred) in enumerate(zip(true_labels, predictions)):
                if true != pred:
                    misclassified.append(
                        {
                            "text": text,
                            "position_index": idx,
                            "true": int(true),
                            "pred": int(pred),
                            "feature_vector": features[idx],
                        }
                    )