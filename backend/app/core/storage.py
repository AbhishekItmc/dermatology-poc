"""
Image storage module for handling patient images
"""
from typing import List, Optional
import numpy as np
import cv2
from pathlib import Path
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ImageStorage(ABC):
    """Abstract base class for image storage"""
    
    @abstractmethod
    async def save_image_set(
        self,
        patient_id: str,
        image_set_id: str,
        images: List[bytes]
    ) -> bool:
        """Save an image set"""
        pass
    
    @abstractmethod
    async def load_image_set(
        self,
        patient_id: str,
        image_set_id: str
    ) -> List[np.ndarray]:
        """Load an image set"""
        pass
    
    @abstractmethod
    async def delete_image_set(
        self,
        patient_id: str,
        image_set_id: str
    ) -> bool:
        """Delete an image set"""
        pass


class LocalImageStorage(ImageStorage):
    """Local filesystem image storage (for development/testing)"""
    
    def __init__(self, base_path: str = "./storage/images"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_patient_dir(self, patient_id: str) -> Path:
        """Get patient directory path"""
        patient_dir = self.base_path / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        return patient_dir
    
    def _get_image_set_dir(self, patient_id: str, image_set_id: str) -> Path:
        """Get image set directory path"""
        image_set_dir = self._get_patient_dir(patient_id) / image_set_id
        image_set_dir.mkdir(parents=True, exist_ok=True)
        return image_set_dir
    
    async def save_image_set(
        self,
        patient_id: str,
        image_set_id: str,
        images: List[bytes]
    ) -> bool:
        """Save an image set to local filesystem"""
        try:
            image_set_dir = self._get_image_set_dir(patient_id, image_set_id)
            
            for idx, image_bytes in enumerate(images):
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.error(f"Failed to decode image {idx}")
                    continue
                
                # Save image
                image_path = image_set_dir / f"image_{idx:03d}.jpg"
                cv2.imwrite(str(image_path), img)
            
            logger.info(f"Saved {len(images)} images for patient {patient_id}, set {image_set_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image set: {e}")
            return False
    
    async def load_image_set(
        self,
        patient_id: str,
        image_set_id: str
    ) -> List[np.ndarray]:
        """Load an image set from local filesystem"""
        try:
            image_set_dir = self._get_image_set_dir(patient_id, image_set_id)
            
            # Find all image files
            image_files = sorted(image_set_dir.glob("image_*.jpg"))
            
            if not image_files:
                logger.warning(f"No images found for patient {patient_id}, set {image_set_id}")
                return []
            
            # Load images
            images = []
            for image_path in image_files:
                img = cv2.imread(str(image_path))
                if img is not None:
                    images.append(img)
            
            logger.info(f"Loaded {len(images)} images for patient {patient_id}, set {image_set_id}")
            return images
            
        except Exception as e:
            logger.error(f"Failed to load image set: {e}")
            return []
    
    async def delete_image_set(
        self,
        patient_id: str,
        image_set_id: str
    ) -> bool:
        """Delete an image set from local filesystem"""
        try:
            image_set_dir = self._get_image_set_dir(patient_id, image_set_id)
            
            # Delete all files in directory
            for file_path in image_set_dir.glob("*"):
                file_path.unlink()
            
            # Delete directory
            image_set_dir.rmdir()
            
            logger.info(f"Deleted image set for patient {patient_id}, set {image_set_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete image set: {e}")
            return False


# Dependency for FastAPI
_storage_instance: Optional[ImageStorage] = None


def get_image_storage() -> ImageStorage:
    """Get image storage instance (dependency injection)"""
    global _storage_instance
    
    if _storage_instance is None:
        # Use local storage for development
        _storage_instance = LocalImageStorage()
    
    return _storage_instance
