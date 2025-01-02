import os
import argparse
from sklearn.model_selection import train_test_split
from models.vibration_cnn import ImprovedVibrationCNN
from models.model_validator import ModelValidator
from utils.saved_model_utils import save_model, load_model

def train_vibration_model(data_path):
    """
    Train the vibration model with cross-validation
    Args:
        data_path: Path to the data directory
    Returns:
        Trained ImprovedVibrationCNN model
    """
    # Initialize model
    model = ImprovedVibrationCNN(max_sequence_length=10000, n_splits=5)

    # Load and preprocess data
    X, y = model.preprocess_data(data_path, augment=True)

    # Train with cross-validation
    model.train_with_cross_validation(X, y, epochs=50, batch_size=32)

    return model

def validate_model(model, X_test, y_test, history=None):
    """
    Comprehensive model validation
    Args:
        model: Trained ImprovedVibrationCNN model
        X_test: Test features
        y_test: Test labels
        history: Training history (optional)
    """
    validator = ModelValidator(model)

    # Get predictions
    y_pred = model.model.predict(X_test)

    # Plot training history if available
    if history is not None:
        print("Plotting training history...")
        validator.plot_training_history(history)

    # Plot confusion matrix and classification report
    print("\nGenerating confusion matrix and classification report...")
    validator.plot_confusion_matrix(y_test, y_pred)

    # Plot ROC curve
    print("\nGenerating ROC curve...")
    validator.plot_roc_curve(y_test, y_pred)

    # Plot Precision-Recall curve
    print("\nGenerating Precision-Recall curve...")
    validator.plot_precision_recall_curve(y_test, y_pred)

    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_scores = validator.analyze_feature_importance(X_test, y_test)

    # Print feature importance scores
    print("\nFeature Importance Scores:")
    for axis, score in importance_scores.items():
        print(f"{axis}: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train vibration model')
    parser.add_argument('--data_path', type=str, default="./data/M01/OPTrain",
                      help='Path to the training data directory')
    parser.add_argument('--save_dir', type=str, default="saved_model",
                      help='Directory to save the model')
    parser.add_argument('--force_retrain', action='store_true',
                      help='Force model retraining even if saved model exists')
    
    args = parser.parse_args()

    # Check if we should train a new model
    should_train = args.force_retrain or not os.path.exists(os.path.join(args.save_dir, 'keras_model.keras'))

    if should_train:
        print("Training new model...")
        # Train model
        model = train_vibration_model(args.data_path)
        # Save the model
        print("Saving model...")
        save_model(model, args.save_dir)
    else:
        print("Loading saved model...")
        model = load_model(args.save_dir)

    # Get test data
    X, y = model.preprocess_data(args.data_path, augment=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Validate model
    validate_model(model, X_test, y_test) 