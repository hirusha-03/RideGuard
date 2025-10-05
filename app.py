import joblib
import pandas as pd
import streamlit as st
from datetime import time
import os

# --------------------------
# 🎯 Load the Trained Model
# --------------------------
MODEL_PATH = "saved_models/Decision_Tree_pipeline.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.warning("Please ensure the file 'Decision_Tree_pipeline.pkl' is placed in the 'saved_models' folder.")
    model = None

# --------------------------
# 🌐 Streamlit Page Config
# --------------------------
st.set_page_config(page_title="RideGuard - Ride Cancellation Prediction", layout="wide")

# --------------------------
# 🎨 Custom CSS Styling
# --------------------------
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

        /* Stylish Predict Button */
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

        /* -------------------- Prediction Result Styling -------------------- */
        .result-cancellation {
            background: linear-gradient(45deg, #ff4f4f, #e63946); /* Strong red gradient */
            border-left: 6px solid #e63946;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(230,57,70,0.2);
            text-align: center;
            color: white !important;
            font-size: 18px !important;
        }

        .result-success {
            background: linear-gradient(45deg, #2e7d32, #388e3c); /* Dark green gradient */
            border-left: 6px solid #2e7d32;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(56,176,0,0.2);
            text-align: center;
            color: white !important;
            font-size: 18px !important;
        }

        .result-cancellation h3, .result-success h3 {
            margin-bottom: 10px;
            font-size: 22px !important;
            font-weight: 700 !important;
        }

        .result-cancellation p, .result-success p {
            font-size: 16px !important;
            font-weight: 500 !important;
        }

        /* Stylish Divider between Results */
        .divider {
            margin: 2rem 0;
            height: 2px;
            background: linear-gradient(90deg, #4e4376, #2b5876);
            border-radius: 2px;
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

# --------------------------
# 🏁 Title Section
# --------------------------
st.markdown("""
    <div class='title-section'>
        <h1>🚖 RideGuard</h1>
        <h5>Predict if a ride is likely to be cancelled based on booking details</h5>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# --------------------------
# 📝 Input Fields
# --------------------------
st.markdown("### 📝 Core Ride Details")
col1, col2= st.columns(2)

with col1:
    pickup_location = st.text_input("Pickup Location", placeholder="e.g., Downtown")
    booking_value = st.number_input("Booking Value ($)", min_value=0.0, step=0.01, format="%.2f")
    ride_distance = st.number_input("Ride Distance (km)", min_value=0.0, step=0.01, format="%.2f")
    hour_time = st.time_input("Pickup Time", time(0, 0))
    hour_of_day = hour_time.hour + hour_time.minute / 60.0
    day_name = st.selectbox("Day of the Week",
                            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_name)
    is_weekend = 1 if day_name in ["Saturday", "Sunday"] else 0

with col2:
    drop_location = st.text_input("Drop Location", placeholder="e.g., Airport Terminal 1")
    payment_method = st.selectbox("Payment Method", ["credit card", "cash", "uber-wallet", "upi", "debit card", "no"])
    vehicle_type = st.selectbox("Vehicle Type", ["premier sedan", "micro", "suv", "go sedan", "auto", "bike", "e-bike"])
    peak_hour = 1 if st.selectbox("Is it Peak Hour?", ["No", "Yes"]) == "Yes" else 0

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------
# ⭐ Customer & Driver Metrics
# --------------------------
st.markdown("### ⭐ Customer & Driver Metrics")
col3, col4, col5, col6 = st.columns(4)

with col3:
    driver_ratings = st.slider("Driver Ratings", 1.0, 5.0, 4.5, 0.1)
with col4:
    customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.5, 0.1)
with col5:
    avg_vtat = st.number_input("Avg VTAT (min)", min_value=0.0, step=0.01, format="%.2f")
with col6:
    avg_ctat = st.number_input("Avg CTAT (min)", min_value=0.0, step=0.01, format="%.2f")

# --------------------------
# 🚦 Prediction Section
# --------------------------
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
predict_button = st.button("🚦 Predict Ride Cancellation")

if predict_button and model:
    user_input = pd.DataFrame([{
        "Booking Value": booking_value,
        "Ride Distance": ride_distance,
        "Driver Ratings": driver_ratings,
        "Vehicle Type": vehicle_type,
        "Pickup Location": pickup_location,
        "Drop Location": drop_location,
        "Payment Method": payment_method,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour_of_day": hour_of_day,
        "peak_hour": peak_hour,
        "Avg VTAT": avg_vtat,
        "Avg CTAT": avg_ctat,
        "Customer Rating": customer_rating
    }])

    # Prediction
    prob = model.predict_proba(user_input)[0][1]
    prediction = int(prob >= 0.4)
    confidence = max(prob, 1 - prob) * 100

    if prediction == 1:
        st.markdown(
            f"<div class='result-cancellation'><h3>❌ Ride Likely CANCELLED</h3><p>Probability of cancellation: {prob:.2f}<br>Confidence: {confidence:.1f}%</p></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='result-success'><h3>✅ Ride Unlikely to be Cancelled</h3><p>Probability of not cancellation: {1 - prob:.2f}<br>Confidence: {confidence:.1f}%</p></div>",
            unsafe_allow_html=True)

elif predict_button and not model:
    st.warning("⚠️ Cannot predict because the model failed to load.")
