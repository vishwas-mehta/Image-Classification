"""
Source package for Image Classification project.

This package contains modules for:
- Data loading and preprocessing (data_loader.py)
- Data augmentation utilities (augmentation.py)
- Model training and evaluation (train.py, evaluate.py)
- Visualization and helper functions (utils.py)
- Custom CNN and ResNet architectures (models/)
"""

__version__ = "1.0.0"
__author__ = "Vishwas Mehta"

from . import data_loader
from . import augmentation
from . import train
from . import evaluate
from . import utils
