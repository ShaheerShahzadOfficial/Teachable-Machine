import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import joblib
import tensorflow as tf
from tensorflow import keras

class Predictor:
    """Handle predictions from all model types"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.class_names = None
        self.target_size = (224, 224)
        
    def load_all_models(self):
        """Load all available trained models"""
        # Load Logistic Regression
        lr_path = self.model_dir / "logistic_regression_model.pkl"
        if lr_path.exists():
            try:
                data = joblib.load(lr_path)
                self.models['Logistic Regression'] = {
                    'model': data['model'],
                    'class_names': data['class_names'],
                    'type': 'sklearn'
                }
                if self.class_names is None:
                    self.class_names = data['class_names']
                print(f"✓ Loaded Logistic Regression model")
            except Exception as e:
                print(f"✗ Error loading Logistic Regression: {e}")
        else:
            print(f"✗ Logistic Regression model not found at {lr_path}")
        
        # Load Random Forest
        rf_path = self.model_dir / "random_forest.pkl"
        if rf_path.exists():
            try:
                data = joblib.load(rf_path)
                self.models['Random Forest'] = {
                    'model': data['model'],
                    'class_names': data['class_names'],
                    'type': 'sklearn'
                }
                if self.class_names is None:
                    self.class_names = data['class_names']
                print(f"✓ Loaded Random Forest model")
            except Exception as e:
                print(f"✗ Error loading Random Forest: {e}")
        else:
            print(f"✗ Random Forest model not found at {rf_path}")
        
        # Load CNN
        cnn_path = self.model_dir / "cnn_keras_model.keras"
        if cnn_path.exists():
            try:
                import json
                model = keras.models.load_model(cnn_path)
                metadata_path = self.model_dir / "cnn_metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.models['CNN (TensorFlow/Keras)'] = {
                    'model': model,
                    'class_names': metadata['class_names'],
                    'type': 'keras'
                }
                if self.class_names is None:
                    self.class_names = metadata['class_names']
                print(f"✓ Loaded CNN model")
            except Exception as e:
                print(f"✗ Error loading CNN: {e}")
        else:
            print(f"✗ CNN model not found at {cnn_path}")
        
        return len(self.models) > 0
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for prediction"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        return img_array
    
    def predict_single(self, image: Image.Image, model_name: str) -> Tuple[str, Dict[str, float]]:
        """Predict using a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_info = self.models[model_name]
        model = model_info['model']
        class_names = model_info['class_names']
        model_type = model_info['type']
        
        # Preprocess image
        img_array = self.preprocess_image(image)
        
        if model_type == 'sklearn':
            # Flatten for sklearn models
            X = img_array.reshape(1, -1)
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
        else:  # keras
            # Add batch dimension
            X = np.expand_dims(img_array, axis=0)
            probabilities = model.predict(X, verbose=0)[0]
            prediction = np.argmax(probabilities)
        
        # Create probability dictionary
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        predicted_class = class_names[prediction]
        
        return predicted_class, prob_dict
    
    def predict_all_models(self, image: Image.Image) -> Dict[str, Tuple[str, Dict[str, float]]]:
        """Get predictions from all loaded models"""
        results = {}
        
        for model_name in self.models.keys():
            try:
                predicted_class, probabilities = self.predict_single(image, model_name)
                results[model_name] = (predicted_class, probabilities)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                results[model_name] = ("Error", {})
        
        return results
    
    def predict_from_webcam_frame(self, frame: np.ndarray) -> Dict[str, Tuple[str, Dict[str, float]]]:
        """Predict from webcam frame"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        
        # Get predictions
        return self.predict_all_models(image)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def is_ready(self) -> bool:
        """Check if predictor has loaded models"""
        return len(self.models) > 0
