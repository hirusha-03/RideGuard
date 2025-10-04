import joblib
import os
import sys

# Define the file path relative to the current script location.
# This assumes 'gradient_boosting_model_pipeline.pkl' is in the 'src/' directory.
model_path = os.path.join(os.path.dirname(__file__), 'gradient_boosting_model_pipeline.pkl')

# Load the complete Pipeline (Pre-processor + Model) once when the module loads
try:
    # 'model' is the full scikit-learn Pipeline object
    model = joblib.load(model_path)
    print(f"Successfully loaded model pipeline from: {model_path}")
except FileNotFoundError:
    print(f"ERROR: Pipeline file not found at: {model_path}")
    print("Please ensure 'gradient_boosting_model_pipeline.pkl' is in the 'src/' directory.")
    # Exit gracefully if the model is essential and missing
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to load model pipeline: {e}")
    sys.exit(1)


def predict(input_data):
    """
    Uses the loaded Pipeline to make a prediction. 
    The Pipeline handles both feature transformation and prediction internally.
    """
    # Check if model loaded successfully before calling predict
    if 'model' not in globals():
        raise RuntimeError("Model pipeline failed to load. Check server logs.")

    return model.predict(input_data)
