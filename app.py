import joblib
import pandas as pd
import streamlit as st
from datetime import time

# Load model
model = joblib.load("saved_models/D_T_model_pipeline.pkl")


# CSS for full-width layout and mobile responsiveness
st.markdown("""
    <style>
        /* Remove default Streamlit padding and margins */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 8rem !important;
            padding-right: 8rem !important;
            max-width: 100% !important;
        }
        
        /* Remove extra spacing */
        .css-1d391kg, .css-18e3th9 {
            padding: 0 !important;
        }
        
        /* Center title section */
        .title-section {
            text-align: center !important;
            margin: 0 1rem !important;
        }
        
        /* Make all inputs larger and full width */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stTimeInput>div>div>input {
            width: 100% !important;
            height: 50px !important;
            font-size: 16px !important;
            padding: 12px !important;
        }
        
        /* Make selectbox larger */
        .stSelectbox>div>div {
            height: 50px !important;
        }
        
        .stSelectbox>div>div>div {
            padding: 12px !important;
            font-size: 16px !important;
        }
        
        /* Make slider larger */
        .stSlider>div>div>div>div {
            height: 8px !important;
        }
        
        .stSlider p {
            font-size: 16px !important;
        }
        
        //* Stylish Predict Button */
.stButton>button {
    width: 100% !important;
    height: 60px !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    color: white !important;
    background: linear-gradient(90deg, #00B4DB, #0078D7)  !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.3s ease-in-out !important;
}

/* Hover effect */
.stButton>button:hover {
    background: linear-gradient(90deg, #00B4DB, #0078D7) !important;
    transform: scale(1.03);
    cursor: pointer;
}

        
        /* Input labels larger */
        .stTextInput>label, .stNumberInput>label, .stSelectbox>label, 
        .stSlider>label, .stTimeInput>label {
            font-size: 16px !important;
            font-weight: 500 !important;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
            
            .title-section h1 {
                font-size: 24px !important;
            }
            
            .title-section h5 {
                font-size: 14px !important;
            }
            
            .stButton>button {
                font-size: 18px !important;
                padding: 0 20px !important;
            }
        }
        
        /* Remove column gaps */
        [data-testid="column"] {
            padding: 0 0.25rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title (centered with small gap on sides)
st.markdown("""
    <div class='title-section'>
        <h1 style='color:#2E86C1;'>🚖 RideGuard - Ride Cancellation Predictor</h1>
        <h5 style='color:gray;'>Predict if a ride is likely to be cancelled or not based on Booking Details</h5>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# First set of inputs in two columns
col1, col2 = st.columns(2, gap="small")

with col1:
    booking_value = st.number_input("Booking Value", min_value=0.0, step=0.01, format="%.2f", placeholder="Enter booking value")
    ride_distance = st.number_input("Ride Distance (km)", min_value=0.0, step=0.01, format="%.2f", placeholder="Enter distance in km")
    driver_ratings = st.slider("Driver Ratings", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    vehicle_type = st.selectbox("Vehicle Type", ["premier sedan", "micro", "suv", "go sedan", "auto", "bike", "e-bike"])
    pickup_location = st.text_input("Pickup Location", placeholder="Enter pickup location")

with col2:
    customer_rating = st.slider("Customer Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    drop_location = st.text_input("Drop Location", placeholder="Enter drop location")
    payment_method = st.selectbox("Payment Method", ["credit card", "cash", "uber-wallet", "upi", "debit card", "no"])
    is_weekend = st.selectbox("Is it Weekend?", ["No", "Yes"], index=0)
    is_weekend = 1 if is_weekend == "Yes" else 0
    hour_time = st.time_input("Select Ride Time", time(0, 0))
    hour_of_day = hour_time.hour + hour_time.minute / 60.0

# Second set of inputs in two columns
col3, col4 = st.columns(2, gap="small")

with col3:
    peak_hour = st.selectbox("Is it Peak Hour?", ["No", "Yes"], index=0)
    peak_hour = 1 if peak_hour == "Yes" else 0
    avg_vtat = st.number_input("Average VTAT (minutes)", min_value=0.0, step=0.01, format="%.2f", placeholder="Enter average VTAT")

with col4:
    avg_ctat = st.number_input("Average CTAT (minutes)", min_value=0.0, step=0.01, format="%.2f", placeholder="Enter average CTAT")

st.markdown("<br>", unsafe_allow_html=True)

# Button on left and prediction result on right
col_result1, col_result2 = st.columns([1, 3])

with col_result1:
    predict_button = st.button("🚦 Predict Ride Cancellation")

with col_result2:

    # Prediction result displayed to the right of button
    if predict_button:
        user_input = pd.DataFrame([{
            "Booking Value": booking_value,
            "Ride Distance": ride_distance,
            "Driver Ratings": driver_ratings,
            "Vehicle Type": vehicle_type,
            "Pickup Location": pickup_location,
            "Drop Location": drop_location,
            "Payment Method": payment_method,
            "day_of_week": pd.Timestamp.now().day_of_week + 1,
            "is_weekend": is_weekend,
            "hour_of_day": hour_of_day,
            "peak_hour": peak_hour,
            "Avg VTAT": avg_vtat,
            "Avg CTAT": avg_ctat,
            "Customer Rating": customer_rating
        }])

        prediction = model.predict(user_input)[0]

        if prediction == 1:
            st.markdown("<h3 style='color: red; font-weight:bold; margin-top: 10px;'>❌ Your Ride is likely to be CANCELLED</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green; font-weight:bold; margin-top: 10px;'>✅ Your Ride is unlikely to be Cancelled</h3>", unsafe_allow_html=True)