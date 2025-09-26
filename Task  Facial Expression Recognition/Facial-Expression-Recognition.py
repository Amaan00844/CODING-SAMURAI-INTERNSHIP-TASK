import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class OAHEGAEmotionRecognizer:
    def __init__(self, img_size=128):
        self.model = None
        self.img_size = img_size
        # OAHEGA dataset specific emotions
        self.emotion_labels = {
            0: 'Happy',
            1: 'Angry', 
            2: 'Sad',
            3: 'Neutral',
            4: 'Surprise',
            5: 'Ahegao'
        }
        print("OAHEGA Emotion Recognition Dataset Initialized")
        print(f"Target emotions: {list(self.emotion_labels.values())}")
    
    def load_dataset_from_csv(self, csv_path="data(1).csv"):
        """Load OAHEGA dataset using the provided CSV file"""
        print(f"Loading dataset from CSV: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return None, None
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"CSV loaded with {len(df)} entries")
        print(f"CSV columns: {list(df.columns)}")
        
        images = []
        labels = []
        
        # Create label mapping
        unique_labels = df.iloc[:, 1].unique()  # Assuming second column is labels
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        print(f"Found emotions: {unique_labels}")
        print(f"Label mapping: {label_map}")
        
        # Update emotion labels to match dataset
        self.emotion_labels = {idx: label for label, idx in label_map.items()}
        
        for idx, row in df.iterrows():
            img_path = row.iloc[0]  # First column is image path
            label = row.iloc[1]  # Second column is label
            
            if os.path.exists(img_path):
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        images.append(img)
                        labels.append(label_map[label])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            else:
                print(f"Image not found: {img_path}")
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32) / 255.0
        y = np.array(labels)
        
        print(f"Dataset loaded: {X.shape[0]} images")
        print(f"Image shape: {X.shape[1:]}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X, y
    
    def visualize_dataset(self, X, y):
        """Visualize the OAHEGA dataset distribution and sample images"""
        unique, counts = np.unique(y, return_counts=True)
        emotion_names = [self.emotion_labels[i] for i in unique]
        
        plt.figure(figsize=(18, 12))
        
        # Dataset distribution
        plt.subplot(2, 3, 1)
        bars = plt.bar(emotion_names, counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
        plt.title('OAHEGA Dataset Distribution')
        plt.xlabel('Emotions')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        # Sample images for each emotion
        for i, (emotion_idx, emotion_name) in enumerate(self.emotion_labels.items()):
            if i < 5:  # Show first 5 emotions
                plt.subplot(2, 3, i + 2)
                
                # Find images for this emotion
                emotion_indices = np.where(y == emotion_idx)[0]
                
                if len(emotion_indices) > 0:
                    # Show first image of this emotion
                    sample_idx = emotion_indices[0]
                    plt.imshow(X[sample_idx])
                    plt.title(f'{emotion_name}\n({len(emotion_indices)} images)')
                    plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed statistics
        print(f"\n📊 OAHEGA Dataset Statistics:")
        print(f"Total images: {len(X)}")
        print(f"Image dimensions: {X.shape[1:]} (RGB)")
        print(f"Number of emotion classes: {len(unique)}")
        print(f"Emotions: {list(emotion_names)}")
        
        for emotion_name, count in zip(emotion_names, counts):
            percentage = (count / len(X)) * 100
            print(f"  • {emotion_name}: {count} images ({percentage:.1f}%)")
    
    def display_sample_grid(self, X, y, samples_per_emotion=4):
        """Display a grid of sample images for each emotion"""
        emotions = list(self.emotion_labels.keys())
        
        fig, axes = plt.subplots(len(emotions), samples_per_emotion, 
                                figsize=(samples_per_emotion * 3, len(emotions) * 2.5))
        
        if len(emotions) == 1:
            axes = [axes]
        
        for emotion_idx, emotion_name in self.emotion_labels.items():
            # Find indices for this emotion
            emotion_indices = np.where(y == emotion_idx)[0]
            
            if len(emotion_indices) > 0:
                # Select random samples
                sample_indices = np.random.choice(
                    emotion_indices, 
                    min(samples_per_emotion, len(emotion_indices)), 
                    replace=False
                )
                
                for j, idx in enumerate(sample_indices):
                    if len(emotions) > 1:
                        ax = axes[emotion_idx][j]
                    else:
                        ax = axes[j]
                    
                    ax.imshow(X[idx])
                    ax.set_title(f'{emotion_name}')
                    ax.axis('off')
                
                # Fill remaining slots if not enough samples
                for j in range(len(sample_indices), samples_per_emotion):
                    if len(emotions) > 1:
                        axes[emotion_idx][j].axis('off')
                    else:
                        axes[j].axis('off')
        
        plt.tight_layout()
        plt.suptitle('OAHEGA Dataset Sample Images', fontsize=16, y=1.02)
        plt.show()
    
    def build_oahega_model(self, num_classes):
        """Build CNN model optimized for OAHEGA dataset"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_size, self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling and Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with appropriate optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train the model with OAHEGA dataset optimizations"""
        num_classes = len(np.unique(y_train))
        
        print(f"🚀 Training OAHEGA model with {num_classes} classes...")
        
        # Build model
        self.model = self.build_oahega_model(num_classes)
        print("Model Architecture:")
        self.model.summary()
        
        # Data augmentation optimized for facial expressions
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_oahega_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Calculate steps
        batch_size = 32
        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)
        
        print(f"Training with batch_size={batch_size}, steps_per_epoch={steps_per_epoch}")
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(X_val, y_val),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot comprehensive training history"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], linewidth=2)
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
        else:
            # Show best accuracy
            best_val_acc = max(history.history['val_accuracy'])
            best_train_acc = max(history.history['accuracy'])
            plt.text(0.5, 0.7, f'Best Validation Accuracy:\n{best_val_acc:.4f}', 
                    transform=plt.gca().transAxes, ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            plt.text(0.5, 0.3, f'Best Training Accuracy:\n{best_train_acc:.4f}', 
                    transform=plt.gca().transAxes, ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            plt.title('Training Summary')
            plt.axis('off')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation for OAHEGA dataset"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return
        
        print("📈 Evaluating OAHEGA model...")
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        emotion_names = [self.emotion_labels[i] for i in range(len(self.emotion_labels))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emotion_names,
                   yticklabels=emotion_names,
                   cbar_kws={'label': 'Number of Predictions'})
        plt.title('OAHEGA Confusion Matrix')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('Actual Emotion')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Classification Report
        print("\n📊 Detailed Classification Report:")
        report = classification_report(y_test, y_pred_classes, 
                                     target_names=emotion_names,
                                     output_dict=True)
        
        # Display as DataFrame for better formatting
        report_df = pd.DataFrame(report).transpose()
        print(report_df.round(4))
        
        # Per-class accuracy
        print(f"\n🎭 Per-Emotion Accuracy:")
        for i, emotion in enumerate(emotion_names):
            if i < len(np.unique(y_test)):
                emotion_mask = y_test == i
                if np.sum(emotion_mask) > 0:
                    emotion_acc = accuracy_score(y_test[emotion_mask], y_pred_classes[emotion_mask])
                    print(f"  • {emotion}: {emotion_acc:.4f} ({emotion_acc*100:.2f}%)")
        
        return accuracy
    
    def predict_emotion(self, image_path):
        """Predict emotion for a single image"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            prediction = self.model.predict(img, verbose=0)
            emotion_idx = np.argmax(prediction)
            confidence = prediction[0][emotion_idx]
            
            emotion_name = self.emotion_labels[emotion_idx]
            
            # Display prediction
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img[0])
            plt.title(f'Predicted: {emotion_name}\nConfidence: {confidence:.2f}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            emotions = list(self.emotion_labels.values())
            probs = prediction[0]
            bars = plt.bar(emotions, probs)
            plt.title('Prediction Probabilities')
            plt.xticks(rotation=45)
            plt.ylabel('Probability')
            
            # Highlight predicted emotion
            bars[emotion_idx].set_color('red')
            
            plt.tight_layout()
            plt.show()
            
            return emotion_name, confidence
            
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return None
    
    def save_model(self, filepath="oahega_emotion_model.h5"):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"💾 Model saved to {filepath}")
        else:
            print("❌ No model to save!")

def main():
    """Main function for OAHEGA dataset processing"""
    print("🎭 OAHEGA Emotion Recognition Dataset Processor")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = OAHEGAEmotionRecognizer(img_size=128)
    
    # Load dataset from CSV
    X, y = recognizer.load_dataset_from_csv()
    
    if X is None or len(X) == 0:
        print("❌ Failed to load dataset!")
        return
    
    # Visualize dataset
    print("\n📊 Visualizing dataset...")
    recognizer.visualize_dataset(X, y)
    recognizer.display_sample_grid(X, y)
    
    # Split dataset
    print("\n🔄 Splitting dataset...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Train model
    print("\n🚀 Training model...")
    history = recognizer.train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # Plot training history
    recognizer.plot_training_history(history)
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    final_accuracy = recognizer.evaluate_model(X_test, y_test)
    
    # Save model
    recognizer.save_model()
    
    print(f"\n✅ OAHEGA Emotion Recognition completed!")
    print(f"🎯 Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"🎭 Emotions recognized: {list(recognizer.emotion_labels.values())}")
    
    # Optional: Test single image prediction
    test_image = input("\n🖼️ Enter path to test image (or press Enter to skip): ").strip()
    if test_image and os.path.exists(test_image):
        result = recognizer.predict_emotion(test_image)
        if result:
            emotion, confidence = result
            print(f"Prediction: {emotion} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()