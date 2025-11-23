import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import joblib
from pathlib import Path
import time

class RandomForestTrainer:
    "Train Random Forest Model"

    def __init__(self,model_save:str = "models"):
        self.model_save_dir = Path(model_save)
        self.model_save_dir.mkdir(exist_ok=True)
        self.model = None
        self.class_names = None

    def train(self,X,y, class_names,progress_callback=None):
        """Train random forest with progress tracking""" 
        self.class_names = class_names
        X_flat = X.reshape(X.shape[0],-1) 
        
        # Split the data
        if progress_callback:
            progress_callback(10,"Splitting dataset")

        X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

        # Train the model

        if progress_callback:
            progress_callback(30,"Training Random Forest Model")

        start_time = time.time()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )    

        # Train with progress updates

        total_trees = 100

        for i in range(0,total_trees,20):
            if i > 0:
                if progress_callback:
                    progress = 30 + int((i/total_trees)*40)
                    progress_callback(progress,f"Training trees{i}/{total_trees}...")

        self.model.fit(X_train,y_train)
        training_time = time.time() - start_time

        if progress_callback:
            progress_callback(70,"Evaluting Model")

        training_accuracy = self.model.score(X_train,y_train)
        testing_accuracy = self.model.score(X_test,y_test)
        y_pred = self.model.predict(X_test)    

        cm = confusion_matrix(y_test,y_pred)

        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        if progress_callback:
            progress_callback(90,"Saving Random Forest Model")

        model_path = self.model_save_dir /"random_forest.pkl" 

        joblib.dump({
            'model': self.model,
            'class_names': self.class_names
        }, model_path)
        
        results = {
            'train_accuracy': training_accuracy,
            'test_accuracy': testing_accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'training_time': training_time,
            'model_path': str(model_path),
            'feature_importance': self.model.feature_importances_
        }
        
        if progress_callback:
            progress_callback(100, "Training complete!")
        
        return results
    
    def load_model(self, model_path: str = None):
        """Load trained model"""
        if model_path is None:
            model_path = self.model_save_dir / "random_forest.pkl"
        
        data = joblib.load(model_path)
        self.model = data['model']
        self.class_names = data['class_names']


    def predict(self,X):
        """Make Predictions"""
        if self.model is None:
            raise ValueError('Model not trained or loaded')

        X_flat = X.reshape(X.shape[0],-1)
        predictions = self.model.predict(X_flat)
        probabilities = self.model.predict_proba(X_flat)
        
        return predictions, probabilities