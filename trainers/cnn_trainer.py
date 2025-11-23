import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers,models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import time
import json


class CNN:
    def __init__(self, model_save_dir: str = "models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.model = None
        self.class_names = []
        self.history = None

    def create_model(self, input: tuple, classes_lenght):
        """Create a CNN model"""
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=input),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                # Second convolutional block
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                # Third convolutional block
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                # Fourth convolutional block
                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                # Flatten and dense layers
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(classes_lenght, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: list,
        epochs: int = 10,
        progress_callback=None,
    ) -> dict:
        """ "Training the Model with progress Tracking"""

        self.class_names = class_names

        # Split the data

        if progress_callback:
            progress_callback(5, "Splitting data into training and validation sets...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=20
        )
        if progress_callback:
            progress_callback(10, "Data split completed.")

        # Building the Model
        input_shape = X_train.shape[1:]
        classes_lenght = len(class_names)
        self.model = self.create_model(input_shape, classes_lenght)

        if progress_callback:
            progress_callback(15, f"Training CNN for {epochs} epochs...")

        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, progress_callback, epochs):
                super().__init__()
                self.progress_callback = progress_callback
                self.epochs = epochs

            def on_echop_end(self, epoch, logs=None):
                progress = 15 + int((epoch + 1) / self.epochs * 65)
                self.progress_callback(
                    progress, f"Epoch {epoch + 1}/{self.epochs} completed."
                )
                acc = logs.get("accuracy")
                val_acc = logs.get("val_accuracy")
                message = f"Epoch {epoch + 1}/{self.epochs} - accuracy: {acc:.4f} - val_accuracy: {val_acc:.4f}"
                self.progress_callback(progress, message)

        callbacks = []
        if progress_callback:
            callbacks.append(ProgressCallback(progress_callback, epochs))

        # Add EarlyStopping callback
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        )

        start_time = time.time()
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0,
        )

        training_time = time.time() - start_time

        if progress_callback:
            progress_callback(80, "Training completed. Evaluating model...")
        # Evaluate

        train_loss, train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)

        # np.argmax is used to get the index of the class with the highest predicted probability
        y_pred = np.argmax(self.model.predict(X_val, verbose=0), axis=1)

        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(
            y_val, y_pred, target_names=class_names, output_dict=True
        )

        if progress_callback:
            progress_callback(95, "Evaluation completed. Saving model...")

        model_file_path = self.model_save_dir / f"cnn_keras_model.keras"

        self.model.save(model_file_path)

        # Save class names
        metadata_path = self.model_save_dir / "cnn_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({"class_names": class_names}, f)

        results = {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(val_accuracy),
            "train_loss": float(train_loss),
            "test_loss": float(val_loss),
            "confusion_matrix": cm,
            "classification_report": report,
            "training_time": training_time,
            "model_path": str(model_file_path),
            "history": {
                "accuracy": [float(x) for x in self.history.history["accuracy"]],
                "val_accuracy": [
                    float(x) for x in self.history.history["val_accuracy"]
                ],
                "loss": [float(x) for x in self.history.history["loss"]],
                "val_loss": [float(x) for x in self.history.history["val_loss"]],
            },
        }
                
        if progress_callback:
            progress_callback(100, "Training complete!")
        
        return results

    def load_model(self, model_path: str = None):
        """Load trained model"""
        if model_path is None:
            model_path = self.model_save_dir / "cnn_model.keras"
        
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = self.model_save_dir / "cnn_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.class_names = metadata['class_names']
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
