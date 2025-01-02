import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import tensorflow as tf

class ModelValidator:
    def __init__(self, model):
        """
        Initialize the validator with a trained model
        Args:
            model: Trained ImprovedVibrationCNN model
        """
        self.model = model

    def plot_training_history(self, history):
        """Plot training metrics history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        # Precision & Recall
        ax3.plot(history.history['precision'], label='Precision')
        ax3.plot(history.history['recall'], label='Recall')
        ax3.set_title('Precision and Recall')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True)

        # AUC
        ax4.plot(history.history['auc'], label='Training AUC')
        ax4.plot(history.history['val_auc'], label='Validation AUC')
        ax4.set_title('Area Under Curve (AUC)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('AUC')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Good', 'Bad']):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, (y_pred > 0.5).astype(int),
                                    target_names=class_names))

    def plot_roc_curve(self, y_true, y_pred):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_pred):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()

    def validate_single_sample(self, sequence):
        """Validate a single vibration sequence"""
        # Preprocess the sequence
        proc_seq = self.model.preprocess_sequence(sequence)
        proc_seq = proc_seq.reshape(1, *proc_seq.shape)

        # Scale the features
        proc_seq = self.model.scaler.transform(proc_seq.reshape(-1, proc_seq.shape[-1]))
        proc_seq = proc_seq.reshape(1, *proc_seq.shape[:-1], -1)

        # Get prediction
        prediction = self.model.model.predict(proc_seq)[0][0]
        predicted_class = "Bad" if prediction > 0.5 else "Good"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        print(f"\nPrediction Results:")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")

        return predicted_class, confidence

    def analyze_feature_importance(self, X, y, n_permutations=10):
        """Analyze feature importance using permutation importance"""
        baseline_score = self.model.model.evaluate(X, y, verbose=0)[0]
        importance_scores = []

        for axis in range(X.shape[2]):
            scores = []
            for _ in range(n_permutations):
                X_permuted = X.copy()
                X_permuted[:, :, axis] = np.random.permutation(X_permuted[:, :, axis])
                score = self.model.model.evaluate(X_permuted, y, verbose=0)[0]
                scores.append(baseline_score - score)
            importance_scores.append(np.mean(scores))

        # Plot feature importance
        plt.figure(figsize=(8, 4))
        axes = ['X-axis', 'Y-axis', 'Z-axis']
        plt.bar(axes, importance_scores)
        plt.title('Feature Importance by Axis')
        plt.xlabel('Vibration Axis')
        plt.ylabel('Importance Score')
        plt.show()

        return dict(zip(axes, importance_scores))