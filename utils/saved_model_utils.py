import os
import joblib
import tensorflow as tf
from models.vibration_cnn import ImprovedVibrationCNN

def save_model(model, save_dir='saved_model'):
    """
    Save the trained model and its components
    """
    os.makedirs(save_dir, exist_ok=True)
    model.model.save(os.path.join(save_dir, 'keras_model.keras'))
    joblib.dump(model.scaler, os.path.join(save_dir, 'scaler.joblib'))
    params = {
        'max_length': model.max_length,
        'n_splits': model.n_splits,
        'random_state': model.random_state
    }
    joblib.dump(params, os.path.join(save_dir, 'params.joblib'))

def load_model(load_dir='saved_model', custom_objects=None):
    """
    Load a trained model and its components
    """
    params = joblib.load(os.path.join(load_dir, 'params.joblib'))
    model = ImprovedVibrationCNN(
        max_sequence_length=params['max_length'],
        n_splits=params['n_splits'],
        random_state=params['random_state']
    )
    
    # Load the Keras model with custom loss function
    model.model = tf.keras.models.load_model(
        os.path.join(load_dir, 'keras_model.keras'),
        custom_objects=custom_objects
    )
    
    model.scaler = joblib.load(os.path.join(load_dir, 'scaler.joblib'))
    return model 