# src/utils.py
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Absolute path to the preprocessor file (must be saved after training)
preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')

# Define the columns (must match the training data)
num_cols = ['booking_value', 'ride_distance', 'driver_ratings', 'customer_rating', 'avg_vtat', 'avg_ctat']
cat_cols = ['vehicle_type', 'pickup_location', 'drop_location', 'payment_method']

# Load the fitted preprocessor
try:
    preprocessor = joblib.load(preprocessor_path)
except FileNotFoundError:
    # Fallback to an unfitted preprocessor (will fail if used) or raise error
    print(f"Warning: Preprocessor not found at {preprocessor_path}. Creating a dummy unfitted one.")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )


def preprocess_data(input_data: pd.DataFrame):
    """
    Preprocess the input data using the loaded, fitted preprocessor:
    - Scale numerical columns
    - One-Hot Encode categorical columns
    """
    # Key change: Use transform() instead of fit_transform()
    transformed_data = preprocessor.transform(input_data)

    # Convert the transformed data back to a DataFrame
    # Need to dynamically get feature names from the loaded preprocessor
    feature_names = num_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

    return transformed_df