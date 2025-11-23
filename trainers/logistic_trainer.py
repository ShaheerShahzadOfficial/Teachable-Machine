import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
import time


class LogisticTrainer:
    """Trainer for Logistic Regression model"""

    def __init__(self, model_save_dir: str = "models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.model = None
        self.class_names = []
        self.history = []

    def train(self, X, Y, classNames, progress_callback=None):
        """Training the logistic regression model"""
        self.class_names = classNames

        X_flat = X.reshape(X.shape[0], -1)

        if progress_callback:
            progress_callback(10, "Splitting data into training and testing sets...")
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, Y, test_size=0.2, random_state=42
        )

        # Train the model
        if progress_callback:
            progress_callback(20, "Training Logistic Regression model...")

        start_time = time.time()
        self.model = LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=42
        )
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
          
        if progress_callback:
            progress_callback(70, "Evaluating model...")  

        # Evaluate the model

        training_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        if progress_callback:
            progress_callback(90, "Saving model...")


        # Save the model

        model_path = self.model_save_dir / "logistic_regression_model.pkl"
        joblib.dump({
            'model': self.model,
            'class_names': self.class_names
        }, model_path)

        results = {
            'train_accuracy': training_accuracy,
            'test_accuracy': test_accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'training_time': training_time,
            'model_path': str(model_path)
        }
        
        if progress_callback:
            progress_callback(100, "Training completed.")

        return results


    def load_model(self, model_path: str):
        """Load a saved logistic regression model"""
        if model_path is None:
            model_path = self.model_save_dir / "logistic_regression_model.pkl"
        
        data = joblib.load(model_path)
        self.model = data['model']
        self.class_names = data['class_names']

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model is not loaded. Please load a model before prediction.")
        
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat)
        probabilities = self.model.predict_proba(X_flat)
        return predictions, probabilities

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
         
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat) 
        probabilities = self.model.predict_proba(X_flat)
        return probabilities