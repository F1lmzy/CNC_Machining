import h5py
import numpy as np
import tensorflow as tf
from utils.saved_model_utils import load_model
from models.model_validator import ModelValidator

def focal_loss(gamma=2., alpha=.25):
    """
    Focal loss for better handling of class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        
        loss_1 = -alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)
        loss_0 = -(1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
        
        return tf.reduce_mean(loss_1 + loss_0)
    
    return focal_loss_fixed

def predict_single_sample(model_path, file_path):
    """
    Make prediction for a single sample
    Args:
        model_path: Path to saved model directory
        file_path: Path to h5 file containing vibration data
    Returns:
        Tuple of (predicted_class, confidence)
    """
    try:
        # Load model with custom loss function
        model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
        validator = ModelValidator(model)

        # Load and process data
        with h5py.File(file_path, 'r') as f:
            vibration_data = f['vibration_data'][:]

        # Make prediction
        predicted_class, confidence = validator.validate_single_sample(vibration_data)

        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

def batch_predict(model_path, file_list):
    """
    Make predictions for multiple samples
    Args:
        model_path: Path to saved model directory
        file_list: List of paths to h5 files
    Returns:
        Dictionary of predictions and confidences
    """
    results = {}
    try:
        # Load model with custom loss function
        model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
        validator = ModelValidator(model)

        for file_path in file_list:
            with h5py.File(file_path, 'r') as f:
                vibration_data = f['vibration_data'][:]
            
            predicted_class, confidence = validator.validate_single_sample(vibration_data)
            results[file_path] = {
                'predicted_class': predicted_class,
                'confidence': confidence
            }
    except Exception as e:
        print(f"Error during batch prediction: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Example usage
    model_path = "saved_model"
    test_file = "./data/M01/OP04/bad/M01_Aug_2019_OP04_000.h5"

    # Single prediction
    predicted_class, confidence = predict_single_sample(model_path, test_file)
    if predicted_class is not None:
        print(f"\nSingle sample prediction:")
        print(f"File: {test_file}")
        print(f"Predicted class: {'Bad' if predicted_class else 'Good'}")
        print(f"Confidence: {confidence:.4f}")
    else:
        print("Prediction failed")

    # # Batch prediction example
    # test_files = [
    #     "./data/M01/OP11/bad/M01_Feb_2021_OP11_000.h5"
    # ]
    #
    # print("\nBatch predictions:")
    # results = batch_predict(model_path, test_files)
    # for file_path, result in results.items():
    #     print(f"\nFile: {file_path}")
    #     print(f"Predicted class: {'Bad' if result['predicted_class'] else 'Good'}")
    #     print(f"Confidence: {result['confidence']:.4f}")