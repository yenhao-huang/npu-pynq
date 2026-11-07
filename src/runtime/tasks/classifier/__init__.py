"""Image classification: the engine, preprocessing and top-k predictions.

    python -m src.runtime.cli classifier --package package --image cat.jpg --labels labels.txt
"""

from src.runtime.tasks.classifier.engine import ImageEngine
from src.runtime.tasks.classifier.preprocess import Prediction, preprocess, top_predictions

__all__ = ['ImageEngine', 'Prediction', 'preprocess', 'top_predictions']
