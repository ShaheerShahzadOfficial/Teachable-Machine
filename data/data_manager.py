import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict

class DatasetManager:
    """Manages image datasets for training ML models"""
    
    def __init__(self, base_dir: str = "data/images"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.min_images_per_class = 10
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        self.target_size = (224, 224)
        
    def validate_image(self, image_file) -> Tuple[bool, str]:
        """Validate uploaded image"""
        try:
            # Check file size
            if hasattr(image_file, 'size') and image_file.size > self.max_image_size:
                return False, f"Image too large. Max size: {self.max_image_size / 1024 / 1024}MB"
            
            # Check format
            img = Image.open(image_file)
            if img.format.lower() not in ['jpeg', 'jpg', 'png', 'bmp']:
                return False, f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
            
            # Check if image can be loaded
            img.verify()
            return True, "Valid image"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def save_images(self, class_name: str, image_files: List) -> Tuple[int, List[str]]:
        """Save uploaded images to class directory"""
        class_dir = self.base_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        errors = []
        
        for idx, image_file in enumerate(image_files):
            try:
                # Validate image
                is_valid, message = self.validate_image(image_file)
                if not is_valid:
                    errors.append(f"File {idx + 1}: {message}")
                    continue
                
                # Reset file pointer after validation
                image_file.seek(0)
                
                # Load and resize image
                img = Image.open(image_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize(self.target_size, Image.LANCZOS)
                
                # Save image
                filename = f"{class_name}_{len(list(class_dir.glob('*.jpg'))) + 1}.jpg"
                img.save(class_dir / filename, 'JPEG', quality=95)
                saved_count += 1
                
            except Exception as e:
                errors.append(f"File {idx + 1}: {str(e)}")
        
        return saved_count, errors
    
    def get_classes(self) -> List[str]:
        """Get list of all classes"""
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def get_class_info(self) -> Dict[str, int]:
        """Get information about each class"""
        info = {}
        for class_name in self.get_classes():
            class_dir = self.base_dir / class_name
            info[class_name] = len(list(class_dir.glob('*.jpg')))
        return info
    
    def delete_class(self, class_name: str) -> bool:
        """Delete a class and all its images"""
        try:
            class_dir = self.base_dir / class_name
            if class_dir.exists():
                shutil.rmtree(class_dir)
            return True
        except Exception as e:
            print(f"Error deleting class: {e}")
            return False
    
    def clear_all_data(self) -> bool:
        """Clear all uploaded data"""
        try:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
                self.base_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False
    
    def reset_all_data(self) -> bool:
        """Reset all data including models and uploaded files"""
        try:
            # Clear uploaded data
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
                self.base_dir.mkdir(exist_ok=True)
            
            # Clear models directory
            models_dir = Path("models")
            if models_dir.exists():
                for model_file in models_dir.glob("*"):
                    if model_file.is_file():
                        model_file.unlink()
            
            return True
        except Exception as e:
            print(f"Error resetting data: {e}")
            return False
    
    def is_ready_for_training(self) -> Tuple[bool, str]:
        """Check if dataset is ready for training"""
        classes = self.get_classes()
        
        if len(classes) < 2:
            return False, "Need at least 2 classes for training"
        
        class_info = self.get_class_info()
        for class_name, count in class_info.items():
            if count < self.min_images_per_class:
                return False, f"Class '{class_name}' needs at least {self.min_images_per_class} images (has {count})"
        
        return True, "Dataset ready for training"
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load all images and labels for training"""
        images = []
        labels = []
        class_names = sorted(self.get_classes())
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.base_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                try:
                    img = Image.open(img_path)
                    img_array = np.array(img) / 255.0  # Normalize
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels), class_names
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        class_info = self.get_class_info()
        total_images = sum(class_info.values())
        
        return {
            'num_classes': len(class_info),
            'total_images': total_images,
            'classes': class_info,
            'ready_for_training': self.is_ready_for_training()[0]
        }
