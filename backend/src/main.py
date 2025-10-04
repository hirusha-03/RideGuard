from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from .model import predict  # Import prediction logic from the model module

app = FastAPI()

# Define the input data schema
class RideData(BaseModel):
    booking_value: float
    ride_distance: float
    driver_ratings: float
    customer_rating: float
    avg_vtat: float
    avg_ctat: float
    vehicle_type: str
    pickup_location: str
    drop_location: str
    payment_method: str

# Define the columns used in your original data for explicit type casting
NUM_COLS = ['booking_value', 'ride_distance', 'driver_ratings', 'customer_rating', 'avg_vtat', 'avg_ctat']
CAT_COLS = ['vehicle_type', 'pickup_location', 'drop_location', 'payment_method']


@app.post("/predict")
def predict_cancellation(data: RideData):
    # Convert Pydantic model to a dictionary
    user_input = data.dict()

    # 1. Convert input to a single-row DataFrame
    # Ensure the column order matches the order used during model training
    user_df = pd.DataFrame([user_input])

    # 2. CRITICAL FIX: Explicitly enforce data types to avoid pipeline errors
    for col in NUM_COLS:
        # Cast to float, using errors='coerce' to turn non-numeric values into NaN (though Pydantic should prevent this)
        user_df[col] = pd.to_numeric(user_df[col], errors='coerce')

    for col in CAT_COLS:
        # Cast to string/object type, which the OneHotEncoder expects
        user_df[col] = user_df[col].astype(str)

    # 3. Call the predict function. The Pipeline handles ALL preprocessing automatically.
    try:
        # The result of predict(user_df) is typically an array of predictions, e.g., [0] or [1]
        prediction = predict(user_df)[0]
    except Exception as e:
        # Catch internal Pipeline/Model errors and return them to the user (500 error will show this message)
        print(f"Prediction Error encountered: {e}")
        return {"error": f"Prediction failed due to internal model processing error. Details: {e}"}

    # 4. Return prediction result
    if prediction == 1:
        return {"prediction": "The ride was canceled by the driver."}
    else:
        return {"prediction": "The ride was not canceled by the driver."}