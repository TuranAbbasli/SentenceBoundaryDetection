from typing import List, Dict, Any, Protocol, Optional, Union
from pathlib import Path
import sklearn.ensemble # type: ignore
import sklearn.metrics # type: ignore


import os
import pickle
import tempfile
from typing import Type, TYPE_CHECKING

from skops.io import dump, load #type: ignore

from constants import DEFAULT_ABBREVIATIONS
from encoders import CharacterEncoder
from features import FeatureExtractor

if TYPE_CHECKING:
    from training import TextSegmenter

ONNX_AVAILABLE = False

class TextSegmentationModel(Protocol):
    """Protocol defining the interface for text segmentation models."""

    def fit(self, X: List[List[int]], y: List[int]) -> None:
        """Fit the model to the data."""
        ...

    def predict(self, X: List[List[int]]) -> List[int]:
        """Predict segmentation labels for the given features."""
        ...

    def get_metrics(self, X: List[List[int]], y: List[int]) -> Dict[str, Any]:
        """Evaluate the model on the given data."""
        ...

    @property
    def is_binary(self) -> bool:
        """Whether the model uses binary classification (boundary/non-boundary)."""
        ...


class BinaryRandomForestModel:
    """
    A text segmentation model based on RandomForest for binary classification.
    Only distinguishes between boundary (1) and non-boundary (0) positions.

    This model supports conversion to ONNX format when the 'onnx' optional
    dependency is installed. ONNX models can be used for faster inference,
    especially when deployed in production environments.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_onnx: bool = False,
        onnx_optimization_level: int = 1,
        **kwargs,
    ):
        """
        Initialize the BinaryRandomForestModel.

        Args:
            threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                        Values below 0.5 favor recall (fewer false negatives),
                                        values above 0.5 favor precision (fewer false positives).
                                        Defaults to 0.5.
            use_onnx (bool, optional): Whether to use ONNX for inference if available.
                                      Defaults to False.
            onnx_optimization_level (int, optional): ONNX optimization level (0-3).
                                                    0: No optimization
                                                    1: Basic optimizations (default)
                                                    2: Extended optimizations
                                                    3: All optimizations including extended memory reuse
                                                    Defaults to 1.
            **kwargs: Parameters to pass to the underlying RandomForestClassifier
        """
        self.threshold = threshold
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.onnx_optimization_level = onnx_optimization_level
        self.onnx_model = None
        self.onnx_session = None
        self.model_params = (
            kwargs.copy()
            if kwargs
            else {
                "n_estimators": 100,
                "max_depth": 16,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "n_jobs": -1,
            }
        )

        # Set class weight to 'balanced' to handle imbalanced data
        if "class_weight" not in self.model_params:
            self.model_params["class_weight"] = "balanced"

        self.model = sklearn.ensemble.RandomForestClassifier(**self.model_params)

        # ONNX related attributes
        self.onnx_model = None
        self.onnx_session = None
        self.feature_count = None

    @property
    def is_binary(self) -> bool:
        """
        Whether the model uses binary classification (boundary/non-boundary).

        Returns:
            bool: Always True for this model
        """
        return True

    def fit(self, X: List[List[int]], y: List[int]) -> None:
        """
        Fit the model to the data.

        Args:
            X (List[List[int]]): Feature vectors
            y (List[int]): Target labels (0 for non-boundary, 1 for boundary)
        """
        # Ensure binary labels
        y_binary = [1 if label > 0 else 0 for label in y]

        # Store feature count for ONNX conversion
        if X and len(X) > 0:
            self.feature_count = len(X[0])

        self.model.fit(X=X, y=y_binary)

        # Convert to ONNX if requested and available
        # if self.use_onnx and ONNX_AVAILABLE:
        #     self.to_onnx()

    def predict(
        self, X: List[List[int]], threshold: Optional[float] = None
    ) -> List[int]:
        """
        Predict segmentation labels for the given features.

        Args:
            X (List[List[int]]): Feature vectors
            threshold (float, optional): Custom probability threshold to use for this prediction.
                                        If None, use the model's default threshold.
                                        Defaults to None.

        Returns:
            List[int]: Predicted labels (0 for non-boundary, 1 for boundary)
        """
        # Use custom threshold if provided, otherwise use the model's default
        thresh = threshold if threshold is not None else self.threshold

        # Use ONNX inference if enabled and available
        # if self.use_onnx and self.onnx_session is not None:
        #     return onnx_predict(self.onnx_session, X, threshold=thresh)

        # Otherwise use scikit-learn inference
        if thresh == 0.5:
            # Use the default scikit-learn prediction for the default threshold
            return self.model.predict(X)
        else:
            # Get class probabilities and apply custom threshold
            probas = self.model.predict_proba(X)
            # Class 1 (boundary) is typically the second column
            return [1 if proba[1] >= thresh else 0 for proba in probas]

    def predict_proba(self, X: List[List[int]]) -> List[List[float]]:
        """
        Predict class probabilities for the given features.

        Args:
            X (List[List[int]]): Feature vectors

        Returns:
            List[List[float]]: Predicted probabilities for each class
        """
        # Use ONNX inference if enabled and available
        # if self.use_onnx and self.onnx_session is not None:
        #     return onnx_predict_proba(self.onnx_session, X)

        # Otherwise use scikit-learn inference
        return self.model.predict_proba(X).tolist()

    def get_metrics(self, X: List[List[int]], y: List[int]) -> Dict[str, Any]:
        """
        Evaluate the model on the given data.

        Args:
            X (List[List[int]]): Feature vectors
            y (List[int]): True labels

        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Convert labels to binary
        y_binary = [1 if label > 0 else 0 for label in y]

        predictions = self.predict(X)

        # Default report structure
        report = {
            "accuracy": sklearn.metrics.accuracy_score(y_binary, predictions),
            "binary_mode": True,
        }

        try:
            # Calculate metrics specific to the boundary class (label=1)
            boundary_precision = sklearn.metrics.precision_score(
                y_binary, predictions, pos_label=1, zero_division=0
            )
            boundary_recall = sklearn.metrics.recall_score(
                y_binary, predictions, pos_label=1, zero_division=0
            )
            boundary_f1 = sklearn.metrics.f1_score(
                y_binary, predictions, pos_label=1, zero_division=0
            )

            # Update report with boundary metrics
            report["precision"] = boundary_precision
            report["recall"] = boundary_recall
            report["f1_score"] = boundary_f1

            # Calculate boundary-specific accuracy
            boundary_indices = [
                i
                for i, (t, p) in enumerate(zip(y_binary, predictions))
                if t == 1 or p == 1
            ]

            if boundary_indices:
                boundary_true = [y_binary[i] for i in boundary_indices]
                boundary_pred = [predictions[i] for i in boundary_indices]
                boundary_accuracy = sklearn.metrics.accuracy_score(
                    boundary_true, boundary_pred
                )
                report["boundary_accuracy"] = boundary_accuracy
            else:
                report["boundary_accuracy"] = 0.0

            # Create full classification report
            full_report = sklearn.metrics.classification_report(
                y_true=y_binary,
                y_pred=predictions,
                target_names=["Non-boundary", "Boundary"],
                labels=[0, 1],
                zero_division=0,
                output_dict=True,
            )

            # Add class-specific metrics
            for k, v in full_report.items():
                if k not in [
                    "accuracy",
                    "macro avg",
                    "weighted avg",
                    "Non-boundary",
                    "Boundary",
                ]:
                    report[f"class_{k}"] = v

        except Exception as e:
            print(f"Warning: Error generating metrics: {e}")
        print(report)
        return report

    def get_feature_importances(self) -> List[float]:
        """
        Get feature importances from the model.

        Returns:
            List[float]: Feature importance scores
        """
        return self.model.feature_importances_.tolist()


def create_model(
    model_type: str = "random_forest",
    threshold: float = 0.5,
    use_onnx: bool = False,
    # onnx_optimization_level: int = 1,
    **kwargs,
) -> TextSegmentationModel:
    """
    Create a text segmentation model.

    Args:
        model_type (str): Type of model to create
                        - "random_forest" or "binary_random_forest": Regular RandomForest model
                        - "feature_selected_rf": RandomForest with feature selection
        threshold (float, optional): Probability threshold for classification (0.0-1.0).
                                   Values below 0.5 favor recall (fewer false negatives),
                                   values above 0.5 favor precision (fewer false positives).
                                   Defaults to 0.5.
        use_onnx (bool, optional): Whether to use ONNX for inference if available.
                                  Requires the 'onnx' optional dependency.
                                  Defaults to False.
        onnx_optimization_level (int, optional): ONNX optimization level (0-3).
                                               0: No optimization
                                               1: Basic optimizations (default)
                                               2: Extended optimizations
                                               3: All optimizations including extended memory reuse
                                               Defaults to 1.
        **kwargs: Parameters to pass to the model constructor

    Returns:
        TextSegmentationModel: A text segmentation model instance

    Raises:
        ValueError: If the model type is not supported
    """
    if model_type.lower() in ["random_forest", "binary_random_forest"]:
        return BinaryRandomForestModel(
            threshold=threshold,
            use_onnx=use_onnx,
            # onnx_optimization_level=onnx_optimization_level,
            **kwargs,
        )
    # elif model_type.lower() in [
    #     "feature_selected_rf",
    #     "feature_selected_random_forest",
    # ]:
    #     # Extract feature selection parameters
    #     feature_selection_threshold = kwargs.pop("feature_selection_threshold", 0.01)
    #     max_features = kwargs.pop("max_features", None)
    #     return FeatureSelectedRandomForestModel(
    #         feature_selection_threshold=feature_selection_threshold,
    #         max_features=max_features,
    #         threshold=threshold,
    #         use_onnx=use_onnx,
    #         onnx_optimization_level=onnx_optimization_level,
    #         **kwargs,
    #     )
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: 'random_forest', 'feature_selected_rf'"
        )


class ModelIO:
    """
    Handles saving and loading segmentation models.
    """

    @staticmethod
    def save(
        segmenter: "TextSegmenter",
        path: str,
        format: str = "skops",
        compress: bool = True,
        compression_level: int = 9,
    ) -> None:
        """
        Save the model and configuration to a file.

        Args:
            segmenter: The segmenter to save
            path (str): Path to save the model
            format (str, optional): Serialization format to use ('skops' or 'pickle').
                                    Defaults to 'skops' for secure serialization.
            compress (bool, optional): Whether to use compression. Defaults to True.
            compression_level (int, optional): Compression level (0-9, where 9 is highest).
                                              Defaults to 9.
        """
        if not segmenter.is_trained:
            raise ValueError("Model has not been trained yet.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save all necessary information to recreate the model
        data = {
            "model": segmenter.model,
            "encoder_cache": segmenter.encoder.cache,
            "config": segmenter.config,
            "version": 5,  # Version for backward compatibility (5 = with compression)
            "compressed": compress,
        }

        # Determine if we need to add a compression extension
        compressed_path = None
        if compress and not (path.endswith(".xz") or path.endswith(".lzma")):
            compressed_path = path + ".xz"

        if format.lower() == "skops":
            # Use skops for secure serialization
            if compress:
                # Create a temporary buffer to hold the serialized data
                import io
                import lzma

                # Create a BytesIO buffer to hold the intermediate result
                buffer = io.BytesIO()

                # Serialize to the buffer using skops
                dump(data, buffer)

                # Get the serialized content
                buffer.seek(0)
                serialized_data = buffer.read()

                # Compress the serialized data using LZMA
                compressed_data = lzma.compress(
                    serialized_data, preset=compression_level
                )

                # Write the compressed data to disk - use compressed_path if specified
                save_path = compressed_path if compressed_path else path
                with open(save_path, "wb") as f:
                    f.write(compressed_data)

                # Remove the uncompressed file if both paths exist and are different
                if compressed_path and os.path.exists(path) and path != save_path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass  # Ignore errors when removing
            else:
                # Regular uncompressed saving
                dump(data, path)
        else:
            # Fallback to pickle format (less secure)
            if compress:
                import lzma

                # Use compressed_path if specified
                save_path = compressed_path if compressed_path else path
                with lzma.open(save_path, "wb", preset=compression_level) as f:
                    pickle.dump(data, f)

                # Remove the uncompressed file if it exists
                if compressed_path and os.path.exists(path) and path != save_path:
                    try:
                        os.remove(path)
                    except Exception:
                        pass  # Ignore errors when removing
            else:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
    @classmethod
    def load(
        cls,
        path: str,
        segmenter_class: Type["TextSegmenter"],
        use_skops: bool = True,
        trust_model: bool = False,
    ) -> "TextSegmenter":
        """
        Load a model and configuration from a file.

        Args:
            path (str): Path to load the model from
            segmenter_class: The TextSegmenter class to instantiate
            use_skops (bool, optional): Whether to use skops to load the model. Defaults to True.
            trust_model (bool, optional): Whether to trust all types in the model file.
                                         Set to True only if you trust the source of the model file.
                                         Defaults to False.

        Returns:
            TextSegmenter: Loaded TextSegmenter instance
        """
        # Check for compression extensions and try alternative paths if needed
        paths_to_try = [path]

        # If the path doesn't end with a compression extension, also try with extensions
        if not (path.endswith(".xz") or path.endswith(".lzma")):
            paths_to_try.append(path + ".xz")
            paths_to_try.append(path + ".lzma")

        # Initialize variables for loading
        data = None
        last_exception = None

        # Try each path until one works
        for try_path in paths_to_try:
            if not os.path.exists(try_path):
                continue

            try:
                # Detect if file is compressed (looking at first few bytes)
                is_compressed = False
                with open(try_path, "rb") as test_file:
                    # LZMA files start with 0xFD, '7', 'z', 'X', 'Z', 0x00
                    file_start = test_file.read(6)
                    if file_start.startswith(b"\xfd7zXZ\x00"):
                        is_compressed = True

                if use_skops:
                    try:
                        if is_compressed:
                            # Handle compressed skops file
                            import io
                            import lzma

                            # Read and decompress the file
                            with open(try_path, "rb") as f:
                                compressed_data = f.read()

                            # Decompress the data
                            decompressed_data = lzma.decompress(compressed_data)

                            # Create a BytesIO buffer with the decompressed data
                            buffer = io.BytesIO(decompressed_data)

                            # Load using skops
                            if trust_model:
                                # Trust all types in the file (use with caution)
                                from skops.io import get_untrusted_types #type: ignore

                                # Need a temporary file to get untrusted types
                                with tempfile.NamedTemporaryFile(
                                    delete=False
                                ) as temp_file:
                                    temp_file.write(decompressed_data)
                                    temp_file.flush()
                                    temp_path = temp_file.name

                                try:
                                    # Get untrusted types from the temp file
                                    untrusted_types = get_untrusted_types(
                                        file=temp_path
                                    )
                                    buffer.seek(0)  # Reset buffer position
                                    data = load(buffer, trusted=untrusted_types)
                                finally:
                                    # Clean up temp file
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                            else:
                                data = load(buffer)

                        else:
                            # Regular uncompressed skops file
                            if trust_model:
                                # Trust all types in the file (use with caution)
                                from skops.io import get_untrusted_types #type: ignore

                                untrusted_types = get_untrusted_types(file=try_path)
                                data = load(try_path, trusted=untrusted_types)
                            else:
                                # Only load trusted types
                                data = load(try_path)
                    except Exception as e:
                        if "UntrustedTypesFoundException" in str(e):
                            # Handle the specific case of untrusted types
                            print(
                                f"Warning: Untrusted types found in model file. "
                                f"Attempting to load with untrusted types: {e}"
                            )

                            # Try to load with untrusted types
                            from skops.io import get_untrusted_types #type: ignore

                            if is_compressed:
                                # Need a temporary file to get untrusted types for compressed file
                                # Get the untrusted types using a temporary file
                                with open(try_path, "rb") as f:
                                    compressed_data = f.read()
                                decompressed_data = lzma.decompress(compressed_data)

                                with tempfile.NamedTemporaryFile(
                                    delete=False
                                ) as temp_file:
                                    temp_file.write(decompressed_data)
                                    temp_file.flush()
                                    temp_path = temp_file.name

                                try:
                                    # Get untrusted types from the temp file
                                    untrusted_types = get_untrusted_types(
                                        file=temp_path
                                    )

                                    # Load with all types trusted (for default model)
                                    buffer = io.BytesIO(decompressed_data)
                                    data = load(buffer, trusted=untrusted_types)
                                finally:
                                    # Clean up temp file
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                            else:
                                untrusted_types = get_untrusted_types(file=try_path)

                                # Register our custom types if possible
                                try:
                                    from skops.io import register_trusted_types #type: ignore

                                    # Import the specific types we need
                                    from models import (
                                        BinaryRandomForestModel,
                                    )
                                    from configs import (
                                        SegmenterConfig,
                                    )

                                    register_trusted_types(BinaryRandomForestModel)
                                    register_trusted_types(SegmenterConfig)
                                except (ImportError, NameError):
                                    pass

                                # Load with all types trusted (for default model)
                                data = load(try_path, trusted=untrusted_types)
                        else:
                            # Re-raise other exceptions
                            raise
                else:
                    # Fallback to pickle (less secure)
                    if is_compressed:
                        import lzma

                        with lzma.open(try_path, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(try_path, "rb") as f:
                            data = pickle.load(f)

                # If we reach here, we successfully loaded the data
                break

            except Exception as e:
                last_exception = e
                continue

        # If we couldn't load from any path, raise the last exception
        if data is None:
            # If all paths fail, try pickle as fallback for backward compatibility
            print(
                f"Warning: Could not load model with specified method: {last_exception}"
            )
            print("Attempting to load with pickle as fallback...")

            for try_path in paths_to_try:
                if not os.path.exists(try_path):
                    continue

                try:
                    # Check if the file might be compressed
                    with open(try_path, "rb") as test_file:
                        file_start = test_file.read(6)

                    if file_start.startswith(b"\xfd7zXZ\x00"):
                        # LZMA compressed file
                        import lzma

                        with lzma.open(try_path, "rb") as f:
                            data = pickle.load(f)
                    else:
                        # Regular file
                        with open(try_path, "rb") as f:
                            data = pickle.load(f)
                    break
                except Exception as e:
                    last_exception = e
                    continue

            if data is None:
                raise ValueError(
                    f"Failed to load model from any of the candidate paths: {paths_to_try}. Last error: {last_exception}"
                )

        encoder = CharacterEncoder()
        encoder.cache = data.get("encoder_cache", {})

        # Handle different versions
        version = data.get("version", 1)

        if version >= 4:
            # Version 4+ uses the config dataclass
            config = data.get("config", None)
        else:
            # Older versions used individual parameters
            from configs import SegmenterConfig

            config = SegmenterConfig(
                left_window=data.get("left_window", 5),
                right_window=data.get("right_window", 5),
                abbreviations=data.get("abbreviations", DEFAULT_ABBREVIATIONS.copy()),
            )

        # Create the feature extractor
        feature_extractor = FeatureExtractor(
            encoder=encoder,
            abbreviations=config.abbreviations,
            use_numpy=config.use_numpy,
            cache_size=config.cache_size,
        )

        segmenter = segmenter_class(
            model=data["model"],
            encoder=encoder,
            feature_extractor=feature_extractor,
            config=config,
        )

        return segmenter
