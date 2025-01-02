import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

class ModelValidator:
    def __init__(self, model):
        self.model = model

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        y_pred_classes = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes))

    def plot_roc_curve(self, y_true, y_pred):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_pred):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def analyze_feature_importance(self, X_test, y_test):
        """Analyze feature importance using perturbation"""
        baseline_score = self.model.model.evaluate(X_test, y_test, verbose=0)[1]
        importance_scores = {}
        
        for i in range(X_test.shape[2]):
            X_perturbed = X_test.copy()
            X_perturbed[:, :, i] = np.random.permutation(X_perturbed[:, :, i])
            perturbed_score = self.model.model.evaluate(X_perturbed, y_test, verbose=0)[1]
            importance = baseline_score - perturbed_score
            importance_scores[f'Axis {i}'] = importance
            
        return importance_scores

    def validate_single_sample(self, sequence):
        """Validate a single sequence"""
        # Preprocess the sequence
        processed_seq = self.model.preprocess_sequence(sequence)
        processed_seq = processed_seq.reshape(1, *processed_seq.shape)
        
        # Scale the sequence
        original_shape = processed_seq.shape
        processed_seq = processed_seq.reshape(-1, processed_seq.shape[-1])
        processed_seq = self.model.scaler.transform(processed_seq)
        processed_seq = processed_seq.reshape(original_shape)
        
        # Get prediction
        prediction = self.model.model.predict(processed_seq)[0][0]
        predicted_class = int(prediction > 0.5)
        confidence = prediction if predicted_class == 1 else 1 - prediction
        
        return predicted_class, confidence 