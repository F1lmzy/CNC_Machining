import numpy as np
import tensorflow as tf
from keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
from utils import data_loader_utils

def focal_loss(gamma=2., alpha=.25):
    """
    Focal loss for better handling of class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate pt
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        # Calculate focal loss
        loss_1 = -alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)
        loss_0 = -(1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
        
        # Return mean loss
        return tf.reduce_mean(loss_1 + loss_0)
    
    return focal_loss_fixed

def calculate_class_weights(y):
    """Calculate balanced class weights"""
    n_samples = len(y)
    n_classes = 2
    counts = np.bincount(y)
    weights = n_samples / (n_classes * counts)
    return {i: weights[i] for i in range(n_classes)}

class ImprovedVibrationCNN:
    def __init__(self, max_sequence_length=10000, n_splits=5, random_state=42):
        self.max_length = max_sequence_length
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.best_model_path = 'best_model.keras'

    def build_model(self, input_shape):
        """Build CNN model with modifications for imbalanced data"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Initial Conv layer with smaller kernel
        x = layers.Conv1D(32, kernel_size=3, padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Add more convolutional layers with skip connections
        for filters in [64, 128]:
            skip = x
            x = layers.Conv1D(filters, kernel_size=3, padding='same',
                             kernel_regularizer=regularizers.l2(0.01))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Conv1D(filters, kernel_size=3, padding='same',
                             kernel_regularizer=regularizers.l2(0.01))(x)
            
            # Add skip connection if shapes match
            if skip.shape[-1] == filters:
                x = layers.Add()([x, skip])
            
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Global attention pooling
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(x.shape[-1])(attention)
        attention = layers.Permute([2, 1])(attention)
        
        x = layers.Multiply()([x, attention])
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with strong regularization
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=focal_loss(),
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        return model

    def preprocess_sequence(self, sequence):
        """Preprocess a single sequence using interpolation"""
        resampled = np.zeros((self.max_length, sequence.shape[1]))
        orig_indices = np.linspace(0, len(sequence) - 1, len(sequence))
        new_indices = np.linspace(0, len(sequence) - 1, self.max_length)
        
        for i in range(sequence.shape[1]):
            resampled[:, i] = np.interp(new_indices, orig_indices, sequence[:, i])
        
        return resampled

    def augment_data(self, sequence, is_bad=False):
        """Apply more aggressive augmentation for bad samples"""
        augmented = []
        # Original sequence
        augmented.append(sequence)
        
        # Number of augmentations for bad samples
        n_augmentations = 8 if is_bad else 1
        
        for _ in range(n_augmentations):
            # Add random noise
            noise_level = np.random.uniform(0.01, 0.03)
            noise = sequence + np.random.normal(0, noise_level, sequence.shape)
            augmented.append(noise)
            
            # Time shift
            shift_percent = np.random.uniform(0.05, 0.15)
            shift = np.roll(sequence, shift=int(len(sequence) * shift_percent), axis=0)
            augmented.append(shift)
            
            # Amplitude scaling
            scale_factor = np.random.uniform(0.9, 1.1)
            scaled = sequence * scale_factor
            augmented.append(scaled)
            
            if is_bad:
                # Additional augmentations for bad samples
                # Time stretching (apply to each channel separately)
                stretched = np.zeros_like(sequence)
                indices = np.linspace(0, len(sequence)-1, len(sequence))
                stretched_indices = indices ** 1.1
                
                # Apply stretching to each channel
                for channel in range(sequence.shape[1]):
                    stretched[:, channel] = np.interp(
                        indices, 
                        stretched_indices, 
                        sequence[:, channel]
                    )
                augmented.append(stretched)
                
                # Combine noise and scaling
                noisy_scaled = scaled + np.random.normal(0, noise_level, sequence.shape)
                augmented.append(noisy_scaled)
        
        return augmented

    def preprocess_data(self, data_path, augment=True, verbose=True):
        """Load and preprocess the vibration data with augmentation"""
        good_path = os.path.join(data_path, "good")
        bad_path = os.path.join(data_path, "bad")
        good_data, good_labels = data_loader_utils.load_tool_research_data(good_path, "good", verbose=verbose)
        bad_data, bad_labels = data_loader_utils.load_tool_research_data(bad_path, "bad", verbose=verbose)

        processed_data = []
        labels = []

        # Process good samples (with less augmentation)
        for sequence in good_data:
            proc_seq = self.preprocess_sequence(sequence)
            if augment:
                augmented = self.augment_data(proc_seq, is_bad=False)
                processed_data.extend(augmented)
                labels.extend([0] * len(augmented))
            else:
                processed_data.append(proc_seq)
                labels.append(0)

        # Process bad samples (with more augmentation)
        for sequence in bad_data:
            proc_seq = self.preprocess_sequence(sequence)
            if augment:
                augmented = self.augment_data(proc_seq, is_bad=True)
                processed_data.extend(augmented)
                labels.extend([1] * len(augmented))
            else:
                processed_data.append(proc_seq)
                labels.append(1)

        X = np.array(processed_data, dtype=np.float32)
        y = np.array(labels)

        # Normalize features
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(original_shape)

        if verbose:
            print(f"Final data shape: {X.shape}")
            print(f"Class distribution - 0s: {np.sum(y == 0)}, 1s: {np.sum(y == 1)}")

        return X, y

    def train_with_cross_validation(self, X, y, epochs=50, batch_size=32):
        """Train model using stratified k-fold cross validation"""
        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                               random_state=self.random_state)
        fold_scores = []

        # Calculate class weights
        class_weights = calculate_class_weights(y)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f'\nFold {fold + 1}/{self.n_splits}')

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

            callbacks = [
                EarlyStopping(
                    monitor='val_auc',
                    patience=10,
                    restore_best_weights=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6
                ),
                ModelCheckpoint(
                    f'model_fold_{fold + 1}.keras',
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max'
                )
            ]

            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weights,  # Add class weights
                verbose=1
            )

            scores = self.model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(scores)

        print("\nCross-validation results:")
        metrics = ['loss'] + self.model.metrics_names
        for i, metric in enumerate(metrics):
            scores = [fold[i] for fold in fold_scores]
            print(f"{metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})") 