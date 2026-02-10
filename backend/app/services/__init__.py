"""
Services package with optional ML dependencies
"""

# Try to import ML-dependent services
ML_AVAILABLE = False
try:
    from .image_preprocessing import ImagePreprocessor
    from .landmark_detection import LandmarkDetector
    from .pigmentation_detection import PigmentationDetector
    from .wrinkle_detection import WrinkleDetector
    from .reconstruction_3d import FacialReconstructor
    from .analysis_service import AnalysisService
    ML_AVAILABLE = True
except ImportError as e:
    # ML dependencies not available - create stub classes
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"ML dependencies not available: {e}")
    
    class ImagePreprocessor:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ML dependencies not installed")
    
    class LandmarkDetector:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ML dependencies not installed")
    
    class PigmentationDetector:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ML dependencies not installed")
    
    class WrinkleDetector:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ML dependencies not installed")
    
    class FacialReconstructor:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ML dependencies not installed")
    
    class AnalysisService:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("ML dependencies not installed")

__all__ = [
    'ImagePreprocessor',
    'LandmarkDetector',
    'PigmentationDetector',
    'WrinkleDetector',
    'FacialReconstructor',
    'AnalysisService',
    'ML_AVAILABLE',
]
