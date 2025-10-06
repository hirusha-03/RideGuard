import joblib
import pandas as pd
import streamlit as st
from datetime import time
import base64
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
# 🖼️ Background Image Setup
# --------------------------
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

add_bg_from_local("bg.jpg")

# --------------------------
# 🎨 Custom CSS Styling
# --------------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 8rem !important;
            padding-right: 8rem !important;
            max-width: 100% !important;
            background: rgba(0, 0, 0, 0.6);  /* Dark overlay for better text contrast */
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }

        .title-section {
            text-align: center !important;
            color: #2E86C1;
        }

        h1, h5 {
            font-family: 'Inter', sans-serif !important;
            color: white !important; /* Ensure heading is white for visibility */
        }

        .stButton>button {
            width: 100%;
            height: 60px;
            font-size: 20px;
            font-weight: 600;
            color: white;
            background: linear-gradient(90deg, #00B4DB, #0078D7);
            border: none;
            border-radius: 10px;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #0098DB, #0068D7);
        }

        .result-cancellation {
            background: linear-gradient(45deg, #ff4f4f, #e63946);
            border-left: 6px solid #e63946;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(230,57,70,0.2);
            text-align: center;
            color: white;
        }

        .result-success {
            background: linear-gradient(45deg, #2e7d32, #388e3c);
            border-left: 6px solid #2e7d32;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(56,176,0,0.2);
            text-align: center;
            color: white;
        }

        .divider {
            margin: 2rem 0;
            height: 2px;
            background: linear-gradient(90deg, #4e4376, #2b5876);
            border-radius: 2px;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# 🏁 Title
# --------------------------
st.markdown("""
    <div class='title-section'>
        <h1>🚖 RideGuard - Ride Cancellation Predictor</h1>
        <h5>Predict if a ride is likely to be cancelled based on booking details</h5>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# --------------------------
# 📝 Input Fields
# --------------------------
st.markdown("### 📝 Core Ride Details")
col1, col2 = st.columns(2)

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
# 🚦 Prediction
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

    prob = model.predict_proba(user_input)[0][1]
    prediction = int(prob >= 0.4)
    confidence = max(prob, 1 - prob) * 100

    if prediction == 1:
        st.markdown(
            f"<div class='result-cancellation'><h3>❌ Ride Likely to be CANCELLED</h3></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='result-success'><h3>✅ Ride Unlikely to be Cancelled</h3></div>",
            unsafe_allow_html=True)

elif predict_button and not model:
    st.warning("⚠️ Cannot predict because the model failed to load.")
