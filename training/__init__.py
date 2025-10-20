"""
Training utilities for the TinyPhysics lateral dynamics simulator.

The modules in this package support:
  * extracting supervised training examples from openpilot logs,
  * defining lightweight PyTorch datasets and models,
  * orchestrating end-to-end training and ONNX export compatible with
    `controls_challenge/tinyphysics.py`.
"""

from __future__ import annotations

__all__ = [
  "data",
  "model",
  "train",
]
