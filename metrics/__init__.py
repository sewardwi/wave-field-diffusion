"""
Sample-quality metrics for wave-field diffusion experiments.

Three modalities, three feature extractors:
  - CIFAR-10  → Inception V3 via clean-fid (standard FID)
  - MNIST     → small 2D CNN classifier (Frechet MNIST Distance + class accuracy)
  - SC09      → small 1D CNN classifier (Frechet SC09 Distance + class accuracy)

Public API:
  from metrics.frechet import frechet_distance, FeatureExtractor
  from metrics.classifier import MNISTClassifier, SC09Classifier
"""
