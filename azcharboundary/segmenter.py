"""
Base text segmentation functionality for the charboundary library.
"""
import time
import random
from functools import lru_cache
import skops.io as sio
import treelite
import treelite.sklearn
from zipfile import ZIP_DEFLATED
from typing import List, Dict, Any, Optional, Union, ClassVar

from azcharboundary.utils.constants import (
    SENTENCE_TAG,
    TERMINAL_SENTENCE_CHAR_LIST,
    PRIMARY_TERMINATORS,
)
from azcharboundary.utils.encoders import CharacterEncoder, CharacterEncoderProtocol
from azcharboundary.utils.features import (
    FeatureExtractor,
    FeatureExtractorProtocol,
    FeatureMatrix,
    PositionLabels,
)
from azcharboundary.utils.models import create_model, TextSegmentationModel

from azcharboundary.utils.types import SegmenterConfig, MetricsResult
from azcharboundary.utils.evaluation import Evaluator

# Segmentation tuning parameters
PATTERN_CONFIDENCE_THRESHOLD = 0.8  # Confidence threshold for pattern matching
CACHE_USE_THRESHOLD = 50  # Number of indices below which to use cached prediction


class TextSegmenter:
    """
    High-level interface for training, saving, loading, and using text segmentation models.

    This simplified implementation only supports binary classification (boundary/non-boundary).

    Key features:
    - Character-level text segmentation
    - Support for sentence and paragraph boundaries
    - Customizable window sizes for context
    - Support for feature selection to improve performance
    - Trained using RandomForest classifiers
    - Support for retrieving character spans for segments

    The segmenter can be used with default parameters or customized for specific needs
    through the configuration parameters, including window sizes, model parameters,
    and feature selection options.
    """

    # Class constants for tag markers
    SENTENCE_TAG: ClassVar[str] = SENTENCE_TAG

    def __init__(
        self,
        model: Optional[TextSegmentationModel] = None,
        encoder: Optional[CharacterEncoderProtocol] = None,
        feature_extractor: Optional[FeatureExtractorProtocol] = None,
        config: Optional[SegmenterConfig] = None,
        prediction_cache_size: int = 10000,
    ):
        """
        Initialize the TextSegmenter.

        Args:
            model (TextSegmentationModel, optional): Model to use.
                If None, a model will be created when training.
            encoder (CharacterEncoderProtocol, optional): Character encoder to use.
                If None, a new one will be created.
            feature_extractor (FeatureExtractorProtocol, optional): Feature extractor to use.
                If None, a new one will be created.
            config (SegmenterConfig, optional): Configuration parameters.
                If None, default configuration will be used.
            prediction_cache_size (int, optional): Size of the prediction cache.
                Larger values use more memory but can improve performance. Defaults to 10000.
        """
        self.config = config or SegmenterConfig()

        self.encoder = encoder or CharacterEncoder()

        self.feature_extractor = feature_extractor or FeatureExtractor(
            encoder=self.encoder,
            abbreviations=self.config.abbreviations,
            use_numpy=self.config.use_numpy,
            cache_size=self.config.cache_size,
        )

        self.model = model
        self.is_trained = model is not None

        # Set up prediction cache
        self.prediction_cache_size = prediction_cache_size
        self._setup_prediction_cache(prediction_cache_size)

    def _setup_prediction_cache(self, cache_size: int) -> None:
        """Set up LRU cache for predictions."""
        # Create cache for single position predictions
        self._cached_predict = lru_cache(maxsize=cache_size)(self._predict_for_position)

    def _predict_for_position(
        self,
        left_context: str,
        right_context: str,
        threshold: float,
    ) -> int:
        """
        Make a prediction for a specific position with context.

        Args:
            text_hash: Hash of the original text (to avoid collisions between different texts)
            position: Position in the original text
            left_context: Text context before and including the character at position
            right_context: Text context after the character at position
            threshold: Probability threshold for classification

        Returns:
            int: Prediction (0 or 1)
        """
        if not self.model:
            return 0

        # Combine contexts
        context = left_context + right_context

        # Calculate the position of the target character in the combined context
        target_pos = len(left_context) - 1

        # Extract features for this position and context
        features = self.feature_extractor.get_char_features(
            context,
            self.config.left_window,
            self.config.right_window,
            positions=[target_pos],
        )

        # Make prediction
        return self.model.predict(features, threshold=threshold)[0]

    def train(
        self,
        data: Union[str, List[str]],
        sample_rate: float = 0.1,
        max_samples: Optional[int] = None,
        model_type: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        left_window: Optional[int] = None,
        right_window: Optional[int] = None,
        num_workers: Optional[int] = None,
        threshold: Optional[float] = None,
        use_feature_selection: bool = False,
        feature_selection_threshold: float = 0.01,
        max_features: Optional[int] = None,
    ) -> MetricsResult:
        """
        Train a new model for text segmentation.

        Args:
            data (Union[str, List[str]]):
                - Path to a training data file
                - List of annotated texts
            sample_rate (float, optional): Rate at which to sample non-terminal positions.
                Defaults to 0.1.
            max_samples (int, optional): Maximum number of samples to process.
                If None, process all samples.
            model_type (str, optional): Type of model to use.
                If None, use the value from config.
            model_params (Dict[str, Any], optional): Parameters for the model.
                If None, use the values from config.
            left_window (int, optional): Size of left context window.
                If None, use the value from config.
            right_window (int, optional): Size of right context window.
                If None, use the value from config.
            num_workers (int, optional): Number of worker processes for parallel processing.
                If None, use the value from config.
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                Values below 0.5 favor recall (fewer false negatives),
                values above 0.5 favor precision (fewer false positives).
                Defaults to None (which means 0.5).
            use_feature_selection (bool, optional): Whether to use feature selection.
                If True, selects important features and retrains the model.
                Defaults to False.
            feature_selection_threshold (float, optional): Importance threshold for selecting features.
                Features with importance below this threshold will be filtered out.
                Only used if use_feature_selection is True.
                Defaults to 0.01.
            max_features (int, optional): Maximum number of features to select.
                If None, use all features above the threshold.
                Only used if use_feature_selection is True.
                Defaults to None.
            use_onnx (bool, optional): Whether to use ONNX for inference if available.
                If True, the model will be converted to ONNX format after training for faster inference.
                Requires the 'onnx' optional dependency.
                Defaults to False.
            onnx_optimization_level (int, optional): ONNX optimization level (0-3) to use.
                0: No optimization
                1: Basic optimizations (default)
                2: Extended optimizations
                3: All optimizations including extended memory reuse
                Only used if use_onnx is True.
                Defaults to None (which uses the default value from config).

        Returns:
            MetricsResult: Training metrics
        """
        # Update config with new values, if provided
        if left_window is not None:
            self.config.left_window = left_window
        if right_window is not None:
            self.config.right_window = right_window
        if num_workers is not None:
            self.config.num_workers = num_workers
        if model_type is not None:
            self.config.model_type = model_type
        if model_params is not None:
            self.config.model_params.update(model_params)
        if threshold is not None:
            self.config.threshold = threshold

        # Store feature selection settings
        self.config.use_feature_selection = use_feature_selection
        self.config.feature_selection_threshold = feature_selection_threshold
        self.config.max_features = max_features

        features: FeatureMatrix = []
        labels: PositionLabels = []

        start = time.time()
        for i, text in enumerate(data):
            if max_samples is not None and i >= max_samples:
                break
            self._process_text_for_training(text, features, labels, sample_rate)
        end = time.time()
        print('Feature extraction finished! Time took: {:.2f}'.format(end-start))

        # Create and train the model
        if self.config.use_feature_selection:
            # Use feature selection model
            print(
                f"Using feature selection with threshold {self.config.feature_selection_threshold}"
            )
            self.model = create_model(
                model_type="feature_selected_rf",
                threshold=self.config.threshold,
                feature_selection_threshold=self.config.feature_selection_threshold,
                max_features=self.config.max_features,
                **(self.config.model_params),
            )
        else:
            # Use regular model
            self.model = create_model(
                model_type=self.config.model_type,
                threshold=self.config.threshold,
                **(self.config.model_params),
            )

        # Print debug info about the training data
        print(f"Training on {len(features)} samples...")
        print(
            f"Window sizes: left={self.config.left_window}, right={self.config.right_window}"
        )
        print(f"Positive samples (boundaries): {labels.count(1)}")
        print(f"Negative samples (non-boundaries): {labels.count(0)}")
        print(f"Positive ratio: {labels.count(1) / len(labels) if labels else 0:.4f}")

        # Fit the model
        self.model.fit(X=features, y=labels)
        self.is_trained = True

        # Print feature selection info if available
        if self.config.use_feature_selection and hasattr(
            self.model, "get_feature_importances"
        ):
            feature_info = self.model.get_feature_importances()
            orig_features = feature_info.get("original_num_features", 0)
            selected_features = feature_info.get("selected_num_features", 0)

            if orig_features > 0:
                print(
                    f"Feature selection reduced dimensions from {orig_features} to {selected_features} features "
                    f"({selected_features / orig_features:.1%} of original)"
                )

                # Print top 10 most important features
                if (
                    "selected_indices" in feature_info
                    and "original_importances" in feature_info
                ):
                    indices = feature_info["selected_indices"][:10]  # Get top 10
                    importances = feature_info["original_importances"]

                    print("\nTop 10 most important features:")
                    for i, idx in enumerate(indices, 1):
                        print(
                            f"  {i}. Feature {idx}: importance={importances[idx]:.4f}"
                        )
                    print("")

        # Evaluate on training data
        report = self.model.get_metrics(features, labels)

        return report

    def _process_text_for_training(
        self,
        text: str,
        features: FeatureMatrix,
        labels: PositionLabels,
        sample_rate: float = 0.1,
    ) -> None:
        """
        Process a text for training and add its features and labels to the provided lists.

        Args:
            text (str): Annotated text
            features (FeatureMatrix): List to which features will be added
            labels (PositionLabels]): List to which labels will be added
            sample_rate (float, optional): Rate at which to sample non-terminal positions.
                Defaults to 0.1.
        """
        clean_text, text_features, text_labels = (
            self.feature_extractor.process_annotated_text(
                text,
                self.config.left_window,
                self.config.right_window,
                self.config.num_workers,
            )
        )

        # Always include terminal characters and a sample of non-terminal characters
        for j, (char, feature_vec, label) in enumerate(
            zip(clean_text, text_features, text_labels)
        ):
            is_terminal = (
                char in TERMINAL_SENTENCE_CHAR_LIST
            )

            # Use modern Python 3.11 pattern matching for cleaner code
            match (label, is_terminal, random.random() < sample_rate):
                case (1, _, _):  # Always include positive samples (boundaries)
                    features.append(feature_vec)
                    labels.append(label)
                case (_, True, _):  # Always include terminal characters
                    features.append(feature_vec)
                    labels.append(label)
                case (_, _, True):  # Sample some non-terminals based on rate
                    features.append(feature_vec)
                    labels.append(label)
                case _:  # Skip other non-terminal characters
                    pass

    def save(
        self,
        path: str,
        serialization_format: str,
        compress: bool = True,
        compression_level: int = 9,
    ) -> None:
        """
        Save the model and configuration to a file.

        Args:
            path (str): Path to save the model.
            serialization_format (str): Format of serialization
            compress (bool, optional): Whether to use compression. Defaults to True.
            compression_level (int, optional): Compression level (0-9, where 9 is highest).
                                              Defaults to 9.
        """
        model = self.model.get_model()

        if serialization_format.lower() == "treelite":        
            tl_model = treelite.sklearn.import_model(model)
            tl_model.save(path)

        else:
            if compress:
                sio.dump(obj=model, file=path, compression=ZIP_DEFLATED, compresslevel=compression_level)
            else:
                sio.dump(obj=model, file=path)

    def load(
        self, path: str, trust_model: bool = False
    ) -> None:
        """
        Load a model and configuration from a file.

        Args:
            path (str): Path to load the model from
            use_skops (bool, optional): Whether to use skops to load the model. Defaults to True.
            trust_model (bool, optional): Whether to trust all types in the model file.
                                         Set to True only if you trust the source of the model file.
                                         Defaults to False.
        """       
        if path.endswith('.skops'):
            if trust_model:
                model = sio.load(file=path, trusted=True)
                self.model.set_model(model)
            else:
                model = sio.load(file=path, trusted=["sklearn.ensemble._forest.RandomForestClassifier"])
                self.model.set_model(model)
            self.is_trained = True
        
        elif path.endswith('.tl'):
            model = treelite.Model.load(path, model_format="treelite")

            self.model.set_inference_predictor(inference_model=model)

            self.is_trained = True
        else:
            print(f"Wrong format model! Path: {path}")

    def inference(self, text: str, threshold: Optional[float] = None):
        """
        Segment text into sentences and paragraphs.

        Args:
            text (str): Text to segment
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            str: Text with sentence annotations
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        # Use the model's threshold if none is provided
        threshold_to_use = threshold if threshold is not None else self.config.threshold

        # Extract features for terminal characters only - optimized approach
        terminal_indices: list[int] = []

        # Pre-identify all terminal characters to batch process them
        for i, char in enumerate(text):
            if char in TERMINAL_SENTENCE_CHAR_LIST:
                terminal_indices.append(i)

        # Skip feature extraction if no terminal characters found
        if not terminal_indices:
            return text
        
        terminal_features = self.feature_extractor.get_char_features(
            text,
            self.config.left_window,
            self.config.right_window,
            positions=terminal_indices,
        )

        if hasattr(self.model, "inference_predict"):
            predictions: list[int] = self.model.inference_predict(
                terminal_features, threshold=threshold_to_use
            )
        else:
            predictions: list[int] = self.model.predict(
                terminal_features, threshold=threshold_to_use
            )

        # Optimization: only create result list if we have boundaries
        if not any(predictions):
            return text
        
        result = list(text)

        tag_shift = 1
        for prediction, terminal_idx in zip(predictions, terminal_indices):
            if prediction:
                result.insert(terminal_idx + tag_shift, SENTENCE_TAG)
                tag_shift += 1

        return "".join(result)

    # Evaluation methods
    def evaluate(
        self, data: Union[str, List[str]], max_samples: Optional[int] = None
    ) -> MetricsResult:
        """
        Evaluate the model on a dataset.

        Args:
            data (Union[str, List[str]]):
                - Path to a test data file
                - List of annotated texts
            max_samples (int, optional): Maximum number of samples to process.
                If None, process all samples.

        Returns:
            MetricsResult: Evaluation metrics
        """
        return Evaluator.evaluate(self, data, max_samples)