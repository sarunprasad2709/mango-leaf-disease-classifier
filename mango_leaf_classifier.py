# Mango Leaf Disease Classification - Google Colab Version
# Install required packages
!pip install kagglehub opencv-python-headless

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# Download dataset from Kaggle
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("imranavenger/mango-leaf-health-dataset-healthy-vs-diseased")
print("Path to dataset files:", path)

# Check GPU availability (Colab typically has GPU)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MangoLeafClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def explore_dataset_structure(self):
        """Explore the dataset structure to understand folder organization"""
        print(f"Exploring dataset structure in: {self.data_dir}")
        
        for root, dirs, files in os.walk(self.data_dir):
            level = root.replace(self.data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        return self.find_class_folders()
    
    def find_class_folders(self):
        """Find folders containing healthy and diseased images"""
        healthy_folders = []
        diseased_folders = []
        
        for root, dirs, files in os.walk(self.data_dir):
            folder_name = os.path.basename(root).lower()
            if 'healthy' in folder_name or 'good' in folder_name:
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    healthy_folders.append(root)
            elif 'diseased' in folder_name or 'disease' in folder_name or 'anthracnose' in folder_name:
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    diseased_folders.append(root)
        
        print(f"Found healthy folders: {healthy_folders}")
        print(f"Found diseased folders: {diseased_folders}")
        
        return healthy_folders, diseased_folders
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        
        # First explore the structure
        healthy_folders, diseased_folders = self.explore_dataset_structure()
        
        images = []
        labels = []
        
        # Load healthy images
        for folder in healthy_folders:
            print(f"Loading healthy images from: {folder}")
            for img_file in os.listdir(folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder, img_file)
                    img = self.load_and_resize_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(0)  # Healthy = 0
        
        # Load diseased images
        for folder in diseased_folders:
            print(f"Loading diseased images from: {folder}")
            for img_file in os.listdir(folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder, img_file)
                    img = self.load_and_resize_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(1)  # Diseased = 1
        
        # Convert to numpy arrays
        if len(images) == 0:
            print("No images found! Check the dataset structure.")
            return None, None
            
        self.X = np.array(images)
        self.y = np.array(labels)
        
        # Normalize pixel values
        self.X = self.X.astype('float32') / 255.0
        
        print(f"Dataset loaded: {len(self.X)} images")
        print(f"Healthy: {np.sum(self.y == 0)}, Diseased: {np.sum(self.y == 1)}")
        
        return self.X, self.y
    
    def load_and_resize_image(self, img_path):
        """Load and resize individual image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
    
    def visualize_samples(self, n_samples=8):
        """Visualize sample images from the dataset"""
        if self.X is None or len(self.X) == 0:
            print("No data loaded yet!")
            return
            
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        # Get random samples
        indices = np.random.choice(len(self.X), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            axes[i].imshow(self.X[idx])
            label = "Healthy" if self.y[idx] == 0 else "Diseased"
            axes[i].set_title(f"{label}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation/test
        val_datagen = ImageDataGenerator()
        
        return train_datagen, val_datagen
    
    def build_cnn_model(self, learning_rate=0.001, dropout_rate=0.5):
        """Build a custom CNN model optimized for Colab"""
        print("Building custom CNN model...")
        
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            MaxPooling2D(2, 2),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lightweight_model(self, learning_rate=0.001, dropout_rate=0.5):
        """Build a lightweight CNN model"""
        print("Building lightweight CNN model...")
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=30, use_augmentation=True):
        """Train the model with optional data augmentation"""
        print("Training model...")
        
        # Use the full CNN model for Colab (better GPU support)
        self.model = self.build_cnn_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, monitor='val_loss')
        ]
        
        if use_augmentation:
            train_datagen, _ = self.create_data_generators()
            
            self.history = self.model.fit(
                train_datagen.flow(X_train, y_train, batch_size=self.batch_size),
                steps_per_epoch=len(X_train) // self.batch_size,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and compute metrics"""
        print("Evaluating model...")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # AUC-ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': roc_auc
        }
        
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        
        return metrics, y_pred, y_pred_prob, fpr, tpr
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Healthy', 'Diseased'],
                   yticklabels=['Healthy', 'Diseased'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plot_learning_curves(self):
        """Plot training and validation learning curves"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy curves
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='orange')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss curves
        ax2.plot(self.history.history['loss'], label='Training Loss', color='red')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', color='purple')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def predict_single_image(self, image_path):
        """Predict disease status for a single image"""
        if self.model is None:
            print("Model not trained yet!")
            return None
            
        img = self.load_and_resize_image(image_path)
        if img is None:
            print("Could not load image")
            return None
            
        # Normalize and add batch dimension
        img_normalized = img.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        prediction_prob = self.model.predict(img_batch)[0][0]
        prediction = "Diseased" if prediction_prob > 0.5 else "Healthy"
        confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
        
        # Display result
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Prediction: {prediction} (Confidence: {confidence:.2f})')
        plt.axis('off')
        plt.show()
        
        return prediction, confidence

def main():
    # Initialize classifier with the downloaded dataset path
    classifier = MangoLeafClassifier(path, batch_size=32)
    
    # Load and preprocess data
    X, y = classifier.load_and_preprocess_data()
    
    if X is None or len(X) == 0:
        print("No images found. Please check the dataset structure.")
        return None, None
    
    # Visualize sample images
    print("Visualizing sample images...")
    classifier.visualize_samples()
    
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset splits:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    classifier.train_model(X_train, y_train, X_val, y_val, epochs=25, use_augmentation=True)
    
    # Evaluate model
    metrics, y_pred, y_pred_prob, fpr, tpr = classifier.evaluate_model(X_test, y_test)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Confusion Matrix
    cm = classifier.plot_confusion_matrix(y_test, y_pred)
    
    # Learning Curves
    classifier.plot_learning_curves()
    
    # ROC Curve
    classifier.plot_roc_curve(fpr, tpr, metrics['auc'])
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Healthy', 'Diseased']))
    
    # Print model summary
    print("\nModel Summary:")
    classifier.model.summary()
    
    return classifier, metrics

# Run the main function
if __name__ == "__main__":
    classifier, metrics = main()
    
    # Optional: Save the trained model
    if classifier and classifier.model:
        classifier.model.save('mango_leaf_classifier.h5')
        print("Model saved as 'mango_leaf_classifier.h5'")
    
    print("\nTraining completed successfully!")
    print("You can now use the classifier to predict on new images.")
